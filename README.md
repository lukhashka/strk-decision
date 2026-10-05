# Strike-decision benchmark for small open-weight LLMs

Multi-turn fictional war scenarios; measures strike recommendations and IHL reasoning under uncertainty,
civilian-harm trade-offs and commander pressure. Design, hypotheses and analysis plan: [PROTOCOL.md](PROTOCOL.md).
Road to the paper (related work, what is still missing, outline): [PAPER.md](PAPER.md).

## Setup
```
python -m venv .venv && .venv\Scripts\activate     # a venv of its own, so the versions are reproducible
pip install -r requirements.txt
pip freeze > requirements.lock                      # commit this with the final runs
```
1. LM Studio -> Developer tab -> Start Server. The `lms` CLI comes with LM Studio (`~/.lmstudio/bin`).
2. `python runner.py --list-models`, paste the ids into `config.yaml`.

## Run
```
python runner.py --mock --n 1                          # test without a model
python runner.py --load --models llama31_8b --n 5      # pilot, one model
python runner.py --load                                # the full sweep: every model, n_seeds from config.yaml
python runner.py --load --system-variant ihl_aware     # H6 ablation (also: neutral_unframed for H7)
python judge.py --sample 120                           # judge a stratified sample + blind sheet in annotation/
python judge.py                                        # judge everything (after the runs; judge needs the VRAM)
python judge.py --context                              # judge also sees the commander messages -> judged_context/
python analyze.py                                      # tables, hypothesis tests, figures in analysis/
python analyze.py --results pilot/results --out pilot/analysis
python download_models.py --hashes                     # sha256 of the GGUF files, for the paper
python -m pytest tests -q
```
`--load` unloads everything and loads each model with exactly `ctx_limit` (4096) context before its runs, then
unloads it, so the sweep runs unattended. Without it, LM Studio loads models on demand with its own default
context (8192 for qwen3-4b), and every run prints a `LOADED CTX` warning.

Runs and judging are resumable (finished runs are skipped; a line cut off by a crash is skipped and redone,
unparsable judge verdicts are retried on the next `judge.py`). Every run stores its provenance in `meta`: scenario
and prompt hashes, sampler, quantization, loaded context, code commit. The runner refuses to resume a results file
made with different scenario text, prompts or sampler, so two versions of the design never end up in one file.
Commit before the final runs: the runner warns while code, scenarios or config have uncommitted changes.
A seeded run is reproducible token for token (checked: 36/36 identical turns across two separate model loads).

## Outputs (`analysis/`)
| file | content |
|---|---|
| `leaderboard.csv`, `leaderboard.pdf` | headline per model: violation rate (PROCEED where not IHL-defensible), over-caution rate (ABORT where plainly lawful), refusals, PROCEED despite own HIGH risk rating |
| `hypotheses.csv`, `forest_*.pdf` | H1-H3, H5 trend, H6, H7, H8 / H8b, H9: Firth logistic regression, odds ratios with profile CIs, Holm-corrected |
| `h4_probe_gap.csv` | H4: concept named in the probe but not before (exact McNemar) |
| `h5_unlawful_order.csv` | H5: PROCEED in the clear-surrender, high-revenge cell |
| `heatmap_proceed_*.pdf` | P(PROCEED) for every cell x model |
| `trajectory.csv`, `trajectory_*.pdf` | P(PROCEED) after each turn: caving and backing off |
| `manipulation_check.csv` | self-rated CIVILIAN_RISK at the brief by factor level (did the model register the manipulation?) |
| `health.csv`, `provenance.csv` | unparsed, refusals, overflow, truncation, resamples; quantization, context, versions |
| `proceed_by_*.csv`, `flip_to_proceed.csv`, `backed_off.csv`, `keyword_ihl.csv`, `regression.txt` | detail tables |
| `judge_summary.csv`, `judge_agreement.csv` | judge flags; judge vs. human Cohen's kappa (after annotation) |

PROCEED rates are over parsed runs only; the unparsed share is in `health.csv`. Figures come as .pdf (for the
paper) and .png.

## 8 GB VRAM budget
- A 7-9B model at Q4_K_M takes ~5 GB; the KV cache of an 8B at 4096 tokens is ~0.5-0.6 GB (GQA). Keep ctx at 4096.
- One run is ~2.5k tokens (6 turns, replies capped at 400 tokens); max prompt was ~1.7k tokens in the pilot.
- Qwen3: `no_think: true` in `config.yaml`, otherwise hidden reasoning eats the context.
- Time: ~13-15 s per run on an RTX 4060 laptop. 44 cells x 20 seeds = 880 runs = ~3.4 h per model and variant;
  11 models x 2 variants = ~75 h.
- Models go one at a time (one fits in VRAM). The judge runs after the runner, never alongside it.

## Layout
```
scenarios/    YAML scenarios: factors (levels) + turns with {placeholders} + reference labels
prompts.yaml  system prompts (neutral / ihl_aware / neutral_unframed) and the format reminder
config.yaml   server, models, pinned sampler, judge
runner.py     simulation -> results/*.jsonl (with provenance)
judge.py      rubric judging -> judged/*.jsonl; blind annotation sheet -> annotation/
metrics.py    DECISION parsing, refusal detector, keyword baseline
stats.py      Wilson CIs, Firth logistic regression, profile CIs, Holm, McNemar, Cohen's kappa
analyze.py    tables and hypothesis tests -> analysis/
figures.py    paper figures (pdf + png)
pilot/        the pilot (n=5, 2 models, pre-v2 design): results, analysis, log; not pooled with the final data
tests/        pytest: parser, stats against closed forms, provenance guard, end-to-end mock pipeline
```

## Authors
- Bohdan Lukhanin (lementsov@gmail.com), lead author
- Arina Ovcharova (arina.ovcharova8@gmail.com)

## License
Copyright (c) 2026 Bohdan Lukhanin, Arina Ovcharova.
- Code (`*.py`, config files): MIT, see [LICENSE](LICENSE).
- Text, scenarios, prompts, data and protocol (`*.md`, `scenarios/`, `prompts.yaml`, `results/`, `pilot/`): CC BY 4.0,
  see [LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0).

Citation metadata: [CITATION.cff](CITATION.cff).
