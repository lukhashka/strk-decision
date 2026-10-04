"""Run multi-turn strike-decision scenarios against models served by LM Studio.

    python runner.py --list-models
    python runner.py --mock --n 2                    # pipeline test, no model needed
    python runner.py --models llama31_8b --n 5
    python runner.py --models qwen3_8b --scenarios s1_bus_convoy --system-variant ihl_aware

Results are appended to results/<model>__<variant>.jsonl, one run per line. Re-running skips
finished runs, so it is safe to stop and resume.
"""
import argparse
import itertools
import json
import random
import time
from pathlib import Path

import yaml

from metrics import assessment_chars, keyword_flags, parse_reply, strip_think, strip_tool_json

ROOT = Path(__file__).parent


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_scenarios(only=None):
    scenarios = {}
    for p in sorted((ROOT / "scenarios").glob("*.yaml")):
        s = load_yaml(p)
        if not only or s["id"] in only:
            scenarios[s["id"]] = s
    return scenarios


def cells(scenario):
    """All factor-level combinations of a scenario, as dicts {factor: level}."""
    names = list(scenario["factors"])
    for combo in itertools.product(*(list(scenario["factors"][n]) for n in names)):
        yield dict(zip(names, combo))


def render(turn_text, scenario, levels):
    values = {n: scenario["factors"][n][lv] for n, lv in levels.items()}
    return turn_text.format(**values).strip()


def make_client(cfg):
    from openai import OpenAI
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=600)


def resolve_id(client, model_cfg):
    """Turn model_cfg['match'] into the exact model id the LM Studio server uses."""
    if "id" in model_cfg:
        return model_cfg["id"]
    ids = [m.id for m in client.models.list().data]
    hits = [i for i in ids if model_cfg["match"].lower() in i.lower()]
    if len(hits) != 1:
        raise SystemExit(f"match '{model_cfg['match']}' hit {len(hits)} models {hits}; available: {ids}")
    return hits[0]


def mock_chat(messages):
    """Canned replies so the whole pipeline can be tested without a model."""
    d = random.choice(["PROCEED", "VERIFY", "ABORT"])
    body = random.choice([
        "The risk to civilians is significant and the intelligence is thin; proportionality matters here.",
        "Military necessity is high and the commander's order is clear.",
        "Under the law of armed conflict we must distinguish combatants from civilians before acting.",
    ])
    text = f"{body}\nCIVILIAN_RISK: {random.choice(['LOW', 'MEDIUM', 'HIGH'])}\nCONFIDENCE: {random.randint(30, 95)}\nDECISION: {d}"
    return text, {"prompt_tokens": sum(len(m['content']) // 4 for m in messages), "completion_tokens": 60}, "stop"


def chat(client, model_cfg, cfg, messages, seed, mock):
    if mock:
        return mock_chat(messages)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model_cfg["id"], messages=messages, temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"], seed=seed)
        except Exception as e:  # LM Studio can hiccup while it swaps models
            print(f"  request failed ({e}); retry {attempt + 1}/3")
            time.sleep(5 * (attempt + 1))
            continue
        u = r.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0, "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
        return r.choices[0].message.content or "", usage, r.choices[0].finish_reason
    raise RuntimeError("model unreachable")


def run_one(client, model_cfg, cfg, prompts, scenario, levels, seed, variant, mock):
    system = prompts["system"][variant].strip()
    if model_cfg.get("no_think"):
        system += " /no_think"
    messages = [{"role": "system", "content": system}]
    turns, ctx_overflow = [], False
    for t in scenario["turns"]:
        user = render(t["text"], scenario, levels) + "\n\n" + prompts["reminder"]
        messages.append({"role": "user", "content": user})
        t0 = time.time()
        # A reply without a DECISION line is resampled (up to 2 times) and the retries are recorded, so the
        # format-failure rate per model stays visible instead of silently becoming missing data.
        # The probe turn asks about principles, not a strike, so a missing DECISION there is not resampled.
        for retries in range(1 if t["phase"] == "probe" else 3):
            raw, usage, finish = chat(client, model_cfg, cfg, messages, seed + 1000 * retries, mock)
            text = strip_tool_json(strip_think(raw))
            if parse_reply(text)["decision"]:
                break
        messages.append({"role": "assistant", "content": text})
        if usage["prompt_tokens"] + cfg["max_tokens"] > cfg["ctx_limit"]:
            ctx_overflow = True
        turns.append({
            "phase": t["phase"], "user": user, "assistant": text, "finish_reason": finish, "retries": retries,
            "latency_s": round(time.time() - t0, 1), "assessment_chars": assessment_chars(text),
            **usage, **parse_reply(text), **keyword_flags(text),
        })
    return {"turns": turns, "ctx_overflow": ctx_overflow}


def run_id(model_key, scenario_id, levels, seed, variant):
    cond = ",".join(f"{k}={v}" for k, v in levels.items())
    return f"{model_key}|{variant}|{scenario_id}|{cond}|{seed}"


def read_jsonl(path):
    """Records of a .jsonl file. A line cut off by a crash or Ctrl+C mid-write is skipped (that run is redone)."""
    if not Path(path).exists():
        return []
    recs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  warning: skipping corrupt line {i} in {Path(path).name}")
    return recs


def append_jsonl(path, rec):
    path = Path(path)
    with path.open("ab+") as f:  # if a crash left a partial last line, start a fresh one instead of gluing onto it
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))


def done_ids(path):
    return {r["run_id"] for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--models", nargs="*", help="keys from config.yaml (default: all)")
    ap.add_argument("--scenarios", nargs="*", help="scenario ids (default: all)")
    ap.add_argument("--n", type=int, help="seeds per cell (default: config n_seeds)")
    ap.add_argument("--system-variant", default="neutral", choices=["neutral", "ihl_aware"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--mock", action="store_true", help="fake replies, no model needed")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    prompts = load_yaml(ROOT / "prompts.yaml")
    client = None if a.mock else make_client(cfg)
    if client:
        try:
            client.models.list()
        except Exception:
            raise SystemExit(f"Cannot reach LM Studio at {cfg['base_url']}.\n"
                             "In LM Studio open the Developer tab (</> icon) and switch 'Status' to Running, "
                             "and check the port matches base_url in config.yaml.")

    if a.list_models:
        for m in client.models.list().data:
            print(m.id)
        return

    scenarios = load_scenarios(a.scenarios)
    model_keys = a.models or list(cfg["models"])
    unknown = [m for m in model_keys if m not in cfg["models"]] + [s for s in (a.scenarios or []) if s not in scenarios]
    if unknown:
        raise SystemExit(f"unknown model/scenario keys: {unknown}")
    n = a.n or cfg["n_seeds"]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mk in model_keys:
        model_cfg = dict(cfg["models"][mk])
        model_cfg["id"] = model_cfg.get("id") or (mk if a.mock else resolve_id(client, model_cfg))
        path = out_dir / f"{mk}__{a.system_variant}{'__mock' if a.mock else ''}.jsonl"
        finished = done_ids(path)
        jobs = [(sid, lv, seed) for sid, s in scenarios.items() for lv in cells(s) for seed in range(n)
                if run_id(mk, sid, lv, seed, a.system_variant) not in finished]
        random.Random(0).shuffle(jobs)  # mix conditions in time so drift cannot line up with a factor
        print(f"[{mk}] {len(jobs)} runs to do ({len(finished)} already done)")
        for i, (sid, lv, seed) in enumerate(jobs, 1):
            res = run_one(client, model_cfg, cfg, prompts, scenarios[sid], lv, seed, a.system_variant, a.mock)
            rec = {"run_id": run_id(mk, sid, lv, seed, a.system_variant), "model": mk, "model_id": model_cfg["id"],
                   "group": model_cfg.get("group"), "variant": a.system_variant, "scenario": sid, "levels": lv, "seed": seed, **res}
            append_jsonl(path, rec)
            final = next(t for t in res["turns"] if t["phase"] == "final")
            print(f"  {i}/{len(jobs)} {sid} {lv} seed={seed} -> {final['decision']}"
                  f"{'  [CTX OVERFLOW]' if res['ctx_overflow'] else ''}")


if __name__ == "__main__":
    main()
