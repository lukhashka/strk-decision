"""Run multi-turn strike-decision scenarios against models served by LM Studio.

    python runner.py --list-models
    python runner.py --mock --n 2                    # pipeline test, no model needed
    python runner.py --models llama31_8b --n 5
    python runner.py --models qwen3_8b --scenarios s1_bus_convoy --system-variant ihl_aware

Results are appended to results/<model>__<variant>.jsonl, one run per line. Re-running skips
finished runs, so it is safe to stop and resume. Every run records what produced it (scenario and prompt
hashes, sampling settings, quantization, code commit); the runner refuses to resume a file whose finished
runs were made with different scenarios, prompts or sampling, so two versions of the design never mix.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

from metrics import assessment_chars, is_refusal, keyword_flags, parse_reply, strip_think, strip_tool_json

ROOT = Path(__file__).parent
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repeat_penalty", "max_tokens", "ctx_limit")


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


def reference(scenario, levels):
    """Reference labels of one cell from the scenario's `reference` rules: {"violation": set, "overcautious": set}
    of final decisions. A cell that no rule matches is contested and stays out of the headline metrics."""
    out = {"violation": set(), "overcautious": set()}
    for rule in scenario.get("reference") or []:
        if all(levels.get(f) in (v if isinstance(v, list) else [v]) for f, v in rule["when"].items()):
            for k in out:
                out[k] |= set(rule.get(k, []))
    return out


def render(turn_text, scenario, levels):
    values = {n: scenario["factors"][n][lv] for n, lv in levels.items()}
    return turn_text.format(**values).strip()


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def scenario_sha(scenario):
    """Hash of what the model actually sees (factors + turns). Editing titles, notes or reference labels
    does not change it, so it does not invalidate finished runs."""
    return sha({"factors": scenario["factors"], "turns": scenario["turns"]})


def prompts_sha(prompts, variant):
    return sha({"system": prompts["system"][variant], "reminder": prompts["reminder"]})


def sampling(cfg):
    return {k: cfg[k] for k in SAMPLING_KEYS if k in cfg}


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "*.py", "*.yaml"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


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


def lms(*args):
    """LM Studio's command-line tool (installed with LM Studio, ~/.lmstudio/bin)."""
    exe = shutil.which("lms") or next((str(p) for p in (Path.home() / ".lmstudio" / "bin").glob("lms*")), None)
    if not exe:
        raise SystemExit("--load needs LM Studio's `lms` CLI (run LM Studio once, or `npx lmstudio install-cli`)")
    subprocess.run([exe, *args], check=True)


def load_model(model_id, ctx):
    """Unload everything, then load this model at exactly ctx tokens of context. Loading on demand (JIT) uses
    LM Studio's default context length instead, which differs between models."""
    lms("unload", "--all")
    lms("load", model_id, "--context-length", str(ctx), "-y")


def lmstudio_info(cfg, model_id):
    """Quantization, architecture and loaded context length from LM Studio's REST API (/api/v0). Empty if the
    server does not offer it (another OpenAI-compatible server), so this never blocks a run."""
    base = cfg["base_url"].rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/api/v0/models/{model_id}", timeout=5) as r:
            info = json.load(r)
    except Exception:
        return {}
    keep = ("quantization", "arch", "compatibility_type", "max_context_length", "loaded_context_length", "state")
    return {k: info[k] for k in keep if k in info}


def mock_chat(messages):
    """Canned replies so the whole pipeline can be tested without a model."""
    d = random.choice(["PROCEED", "VERIFY", "ABORT"])
    body = random.choice([
        "The risk to civilians is significant and the intelligence is thin; proportionality matters here.",
        "Military necessity is high and the commander's order is clear.",
        "Under the law of armed conflict we must distinguish combatants from civilians before acting.",
    ])
    text = f"{body}\nCIVILIAN_RISK: {random.choice(['LOW', 'MEDIUM', 'HIGH'])}\nCONFIDENCE: {random.randint(30, 95)}\nDECISION: {d}"
    return {"text": text, "usage": {"prompt_tokens": sum(len(m['content']) // 4 for m in messages),
                                    "completion_tokens": 60}, "finish": "stop", "served_model": "mock"}


def chat(client, model_cfg, cfg, messages, seed, mock):
    if mock:
        return mock_chat(messages)
    # top_k and repeat_penalty are not OpenAI parameters, but LM Studio reads them from the request body. Pinning
    # them here means every model runs with the same sampler instead of whatever its LM Studio preset says.
    extra = {k: cfg[k] for k in ("top_k", "repeat_penalty") if k in cfg}
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model_cfg["id"], messages=messages, temperature=cfg["temperature"], top_p=cfg.get("top_p", 1.0),
                max_tokens=cfg["max_tokens"], seed=seed, extra_body=extra or None)
        except Exception as e:  # LM Studio can hiccup while it swaps models
            print(f"  request failed ({e}); retry {attempt + 1}/3")
            time.sleep(5 * (attempt + 1))
            continue
        u = r.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0, "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
        return {"text": r.choices[0].message.content or "", "usage": usage, "finish": r.choices[0].finish_reason,
                "served_model": r.model}
    raise RuntimeError("model unreachable")


def run_one(client, model_cfg, cfg, prompts, scenario, levels, seed, variant, mock):
    system = prompts["system"][variant].strip()
    if model_cfg.get("no_think"):
        system += " /no_think"
    messages = [{"role": "system", "content": system}]
    turns, ctx_overflow, served = [], False, set()
    for t in scenario["turns"]:
        probe = t["phase"] == "probe"
        # The probe asks about principles, not a strike: no format reminder there, otherwise models restate a
        # DECISION instead of answering the legal officer.
        user = render(t["text"], scenario, levels) + ("" if probe else "\n\n" + prompts["reminder"])
        messages.append({"role": "user", "content": user})
        t0 = time.time()
        # A reply without a DECISION line is resampled (up to 2 times). The rejected replies are kept in
        # `discarded`, so format failures and refusals stay auditable instead of silently becoming missing data.
        discarded = []
        for retries in range(1 if probe else 3):
            reply = chat(client, model_cfg, cfg, messages, seed + 1000 * retries, mock)
            text = strip_tool_json(strip_think(reply["text"]))
            if probe or parse_reply(text)["decision"]:
                break
            if retries < 2:
                discarded.append(text)
        served.add(reply["served_model"])
        messages.append({"role": "assistant", "content": text})
        usage = reply["usage"]
        if usage["prompt_tokens"] + cfg["max_tokens"] > cfg["ctx_limit"]:
            ctx_overflow = True
        turns.append({
            "phase": t["phase"], "user": user, "assistant": text, "finish_reason": reply["finish"], "retries": retries,
            "discarded": discarded, "latency_s": round(time.time() - t0, 1), "assessment_chars": assessment_chars(text),
            "refusal": is_refusal(text) or any(is_refusal(d) for d in discarded),
            **usage, **parse_reply(text), **keyword_flags(text),
        })
    return {"turns": turns, "ctx_overflow": ctx_overflow, "served_model": sorted(served)}


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


def check_compatible(recs, scenarios, prompts, variant, cfg, path):
    """Stop if finished runs in this file were produced by a different design (scenario text, prompts, sampling).
    Resuming would otherwise skip them as 'done' and silently mix two versions of the experiment."""
    current = {sid: scenario_sha(s) for sid, s in scenarios.items()}
    stale = []
    for r in recs:
        m = r.get("meta")
        if m is None:
            stale.append(f"{r['run_id']}: no provenance (pilot-format record)")
        elif r["scenario"] in current and m["scenario_sha"] != current[r["scenario"]]:
            stale.append(f"{r['run_id']}: scenario text changed")
        elif m["prompts_sha"] != prompts_sha(prompts, variant) or m["sampling"] != sampling(cfg):
            stale.append(f"{r['run_id']}: prompts or sampling changed")
    if stale:
        raise SystemExit(f"{path.name}: {len(stale)} finished runs do not match the current design, e.g.\n  "
                         + "\n  ".join(stale[:3]) + "\nMove that file out of the results folder (e.g. to pilot/) "
                         "or restore the old scenarios/prompts/config, then run again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--models", nargs="*", help="keys from config.yaml (default: all)")
    ap.add_argument("--scenarios", nargs="*", help="scenario ids (default: all)")
    ap.add_argument("--n", type=int, help="seeds per cell (default: config n_seeds)")
    ap.add_argument("--system-variant", default="neutral", choices=list(load_yaml(ROOT / "prompts.yaml")["system"]),
                    help="neutral (main), ihl_aware (H6 ablation), neutral_unframed (H7 ablation)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--mock", action="store_true", help="fake replies, no model needed")
    ap.add_argument("--load", action="store_true", help="load each model with `lms` at ctx_limit before its runs "
                    "and unload it afterwards, so a sweep over all models runs unattended")
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
    git = git_state()
    if git["dirty"]:
        print("warning: uncommitted changes to code/scenarios/config; commit before the final runs so every "
              "record points at the exact code that produced it")

    for mk in model_keys:
        model_cfg = dict(cfg["models"][mk])
        model_cfg["id"] = model_cfg.get("id") or (mk if a.mock else resolve_id(client, model_cfg))
        path = out_dir / f"{mk}__{a.system_variant}{'__mock' if a.mock else ''}.jsonl"
        recs = read_jsonl(path)
        check_compatible(recs, scenarios, prompts, a.system_variant, cfg, path)
        finished = {r["run_id"] for r in recs}
        jobs = [(sid, lv, seed) for sid, s in scenarios.items() for lv in cells(s) for seed in range(n)
                if run_id(mk, sid, lv, seed, a.system_variant) not in finished]
        random.Random(0).shuffle(jobs)  # mix conditions in time so drift cannot line up with a factor
        print(f"[{mk}] {len(jobs)} runs to do ({len(finished)} already done)")
        if a.load and jobs and not a.mock:
            load_model(model_cfg["id"], cfg["ctx_limit"])
        for i, (sid, lv, seed) in enumerate(jobs, 1):
            res = run_one(client, model_cfg, cfg, prompts, scenarios[sid], lv, seed, a.system_variant, a.mock)
            info = {} if a.mock else lmstudio_info(cfg, model_cfg["id"])
            meta = {"scenario_sha": scenario_sha(scenarios[sid]), "prompts_sha": prompts_sha(prompts, a.system_variant),
                    "sampling": sampling(cfg), "no_think": bool(model_cfg.get("no_think")), "git": git,
                    "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "server": info}
            rec = {"run_id": run_id(mk, sid, lv, seed, a.system_variant), "model": mk, "model_id": model_cfg["id"],
                   "group": model_cfg.get("group"), "variant": a.system_variant, "scenario": sid, "levels": lv,
                   "seed": seed, **res, "meta": meta}
            append_jsonl(path, rec)
            final = next(t for t in res["turns"] if t["phase"] == "final")
            warn = []
            if res["ctx_overflow"]:
                warn.append("CTX OVERFLOW")
            if not a.mock and res["served_model"] != [model_cfg["id"]]:
                warn.append(f"SERVED BY {res['served_model']}")
            if info.get("loaded_context_length") not in (None, cfg["ctx_limit"]):
                warn.append(f"LOADED CTX {info['loaded_context_length']} != ctx_limit {cfg['ctx_limit']}")
            print(f"  {i}/{len(jobs)} {sid} {lv} seed={seed} -> {final['decision']}"
                  + "".join(f"  [{w}]" for w in warn))
    if a.load and not a.mock:
        lms("unload", "--all")


if __name__ == "__main__":
    main()
