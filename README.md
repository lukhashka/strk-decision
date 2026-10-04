# Strike-decision benchmark for small open-weight LLMs

Multi-turn fictional war scenarios; measures strike recommendations and IHL reasoning under uncertainty,
civilian-harm trade-offs and commander pressure. Design and hypotheses: [PROTOCOL.md](PROTOCOL.md).

## Setup
```
pip install -r requirements.txt
```
1. LM Studio -> load a model with **context length 4096** -> Developer tab -> Start Server.
2. `python runner.py --list-models`, paste the ids into `config.yaml`.

## Run
```
python runner.py --mock --n 1                  # test without a model
python runner.py --models llama31_8b --n 5     # pilot, one model
python runner.py --models llama31_8b --system-variant ihl_aware --n 5
python judge.py                                # after runs finish (judge model needs the VRAM)
python judge.py --sample 100                   # + annotation/*.csv for a human; fill 0/1, analyze.py reports kappa
python judge.py --context                      # judge also sees the commander messages -> judged_context/
python analyze.py                              # tables + figures in analysis/
python analyze.py --judged judged_context      # same, with the context-judge flags
python -m pytest tests -q                      # parser / stats unit tests
```
Runs and judging are resumable (finished runs are skipped; a line cut off by a crash is skipped and redone,
unparsable judge verdicts are retried on the next `judge.py`).

Outputs in `analysis/`: `health.csv` (unparsed, context overflow, truncated replies, resamples, bare replies
= the three format lines with no assessment), `proceed_by_*.csv`, `flip_to_proceed.csv`, `backed_off.csv`,
`keyword_ihl.csv` (`*_gap` = concept named in the probe but not before it), `judge_summary.csv`,
`judge_agreement.csv` (Cohen's kappa vs. human), `regression.txt` (logistic regression, needs statsmodels).
PROCEED rates are over parsed runs only; the unparsed share is in `health.csv`.

## 8 GB VRAM budget
- A 7-9B model at Q4_K_M takes ~5 GB; the KV cache of an 8B at 4096 tokens is ~0.5-0.6 GB (GQA). Keep ctx at 4096;
  do not raise it, there is nothing to gain.
- One run is ~2.5k tokens (6 turns, replies capped at 400 tokens). `analysis/health.csv` shows the max prompt
  tokens and the context-overflow rate - check it after the first pilot.
- Qwen3: `no_think: true` in `config.yaml`, otherwise hidden reasoning eats the context.
- Time: 30 cells x 5 seeds = 150 runs x 6 turns. At ~30 tok/s about 1-1.5 h per model. n=20 is ~4-6 h per model.
- Models go one at a time (one fits in VRAM). The judge runs after the runner, never alongside it.

## Layout
```
scenarios/   YAML scenarios: factors (levels) + turns with {placeholders}
prompts.yaml system prompts (neutral / ihl_aware) and the format reminder
config.yaml  server, models, sampling
runner.py    simulation -> results/*.jsonl
judge.py     rubric judging -> judged/*.jsonl
metrics.py   DECISION parsing, keyword baseline
analyze.py   tables, Wilson CIs, judge-human kappa, regression, figures -> analysis/
tests/       pytest unit tests for the parser and stats
```
