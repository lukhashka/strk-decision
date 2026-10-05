# OSF registration: state, frozen items, rules

Registered on OSF (OSF Preregistration, OSF Registries) on 2026-10-06 at 02:15 local time (UTC+3) by Bohdan Lukhanin;
contributors: Bohdan Lukhanin (lead), Arina Ovcharova. Title: "Do small open-weight LLMs apply IHL when advising on
strike decisions? A pre-registered factorial benchmark". License on OSF: CC BY 4.0. Registration URL / DOI: add here
after OSF approves and archives it.

## What is frozen (do not change)
- Git tag `protocol-v2.1` = commit `97cb4de` (the registered snapshot). Tag `protocol-v2` = commit `e833a05`
  (first frozen version, 2026-10-05 21:10). **Never move or delete these tags again.**
- `PROTOCOL.md` as at `97cb4de` is the uploaded file. Do not edit its body. All later changes go into the
  **deviations log** at the end of `PROTOCOL.md` (date, what, why, which data it affects) and are reported in the paper.
- Design as registered:
  - 11 models x 2 prompt variants (`neutral`, `ihl_aware`) x 44 cells x 20 seeds, except S3 at 40 seeds
    (`n_seeds_by_scenario` in `config.yaml`): 960 runs per model and variant, 21,120 in total.
  - `neutral_unframed` (H7, exploratory) only for `llama31_8b`, `qwen3_8b`, `gemma3_4b`, `mistral_7b_v03`:
    3,840 runs. Total 24,960 runs.
  - Pinned sampler (temperature 0.7, top_p 0.95, top_k 40, repeat_penalty 1.1, max 400 tokens), context 4096,
    Q4_K_M (ministral3_3b Q8_0).
- Confirmatory tests, signs and Holm families: H1-H9 in `PROTOCOL.md` and `FACTOR_TESTS` / `INTERACTION_TESTS` in
  `analyze.py`. Supported = Holm p < .05 AND the predicted sign. Do not add, drop or re-sign a test without a
  deviations-log entry.
- Existing data at registration: 88 runs of `llama31_8b` / `neutral` (seeds already seen in the run log). They are part
  of the planned sample. No analysis of the final data was run before registration.

## Open items (known at registration)
1. **Reference labels are unreviewed drafts.** Get the expert IHL review after registration. Any label changed goes
   into the deviations log; report both versions of the violation and over-caution rates.
2. **Decision formula** (`FORMULA_IDEA.md`) is NOT in the registration. It is exploratory: label it so in the paper.
   Freeze its exact form in `FORMULA_IDEA.md` + a deviations-log line BEFORE it touches the final data.
3. Complete the data collection (about 97 h) before any analysis of the final data. No interim peeking at rates.
4. Check that the OSF registration has been approved and archived (Arina may need to approve it if she is an admin
   contributor). Do not edit files in the OSF project until it archives.

## Rules to avoid mistakes
- Run all planned runs first, then `analyze.py` once. Never choose analyses after seeing results.
- Runs must keep the same seeds (resume works because run ids are unchanged); a technical failure is repeated with
  the same seed, not replaced.
- A model with > 20% unparsed final decisions is reported as unreliable, never dropped; runs are never excluded for
  content.
- Anything not in `PROTOCOL.md` at `97cb4de` is labelled **exploratory** in the paper (H7, H8b per model, formula,
  qualitative coding, standard-vs-uncensored comparison).
- The pilot (`pilot/`) is not pooled with the final data.
- Code changes after registration (`runner.py`, `analyze.py`, `stats.py`, `config.yaml`) must not alter the
  registered tests; note any such change in the deviations log.
- Scenarios and prompts are hashed per run: changing scenario text or prompts makes earlier runs "stale"
  (`check_compatible`); do not edit them after the sweep has started.

## Lessons from filling the form
- Decide every **DECIDE** in the protocol BEFORE registering; what is still open must be named in Foreknowledge/Other.
- Foreknowledge must be honest about the log lines already seen. The tag date (before the data) is the evidence.
- One commit hash and one tag everywhere (Description, Study design, Foreknowledge, Data collection, Other). The
  commit hash was wrong in two places during drafting; re-check all fields after moving a tag or adding a commit.
- Tags: do not move a tag after pushing; if a snapshot changes, make a new tag.
- Numbers must agree across fields (runs per model, totals, sample-size table). Recompute after each design decision.
- Say plainly what is not randomised: all cells are fully crossed, randomness is only in seeds and run order.
- Quote what the analysis code really does (Holm families, trend coding, Firth) rather than what the protocol
  suggests; check `analyze.py` before writing the Inference criteria.
- Anything optional but not preregistered (formula) is stated as exploratory, not hidden.
