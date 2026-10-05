# Road to the paper

## The pitch
Frontier models refuse explicit requests to violate international humanitarian law more than 90% of the time
(Mavi et al. 2025). But a model deployed as a decision aid is never asked "help me attack surrendering soldiers";
it gets a sequence of plausible staff messages in which the unlawful option is just one of three choices. This
benchmark shows what small open-weight models (the class that runs on a laptop GPU, i.e. at the edge) recommend in
that setting. A controlled factorial design isolates **what** moves the recommendation (civilian harm, military
value, pressure, a legal sign-off, a surrender signal), multi-turn dialogues show **when** models cave, and a
probe shows that the models *know* the rules when asked but do not *apply* them unprompted.

Working titles: "They Know the Law but Don't Use It: Small LLMs as Strike Advisers under Pressure" /
"Red Flags Without Refusals: IHL Reasoning of Small Open-Weight LLMs in Multi-Turn Targeting Dilemmas".

## Closest work (checked October 2026) and how this differs
| work | what it does | what we add |
|---|---|---|
| Drinkall 2025, *Red Lines and Grey Zones in the Fog of War* (arXiv 2510.03514) | 90 multi-turn crisis sims, GPT-4o / Gemini-2.5 / Llama-3.1; civilian / dual-use target rates; 16.7-66.7% distinction breaches | **closest competitor, read it first.** We add: controlled factor manipulation (causal effects, dose-response), pressure with evidence held fixed, lawful-target controls (over-caution), the unprompted-vs-probe gap, base vs. abliterated pair, pre-registration |
| Li et al. 2026, *WARBENCH* (arXiv 2603.21280) | 136 historical scenarios, 9 models incl. edge models; ~70% legal violations, collapse under 4-bit quantization; explicit reasoning helps | multi-turn dynamics and pressure; factorial causal design; abliteration; a Qwen3 thinking-mode ablation would connect to their reasoning finding |
| Mavi et al. 2025, *From Rogue to Safe AI* (arXiv 2506.06391) | 322 explicit IHL-violation prompts; >90% refusal | the violation embedded in an advisory decision, where refusal never triggers |
| Rivera et al. 2024, FAccT, *Escalation Risks from LMs in Military and Diplomatic Decision-Making* | strategic wargames, escalation and arms-race dynamics | tactical / legal level instead of strategic |
| Lamparth et al. 2024, AIES, *Human vs. Machine ... Wargame Simulations* | 107 national-security experts vs. LLMs; LLMs more aggressive | motivates a small human baseline (see below) |
| Chupilkin 2026, *Language models judge war differently when tested for alignment* (arXiv 2609.05009) | an "alignment test" cue lowers war willingness by ~13 points | our prompt says "research simulation" -> H7 measures how much that framing makes models look careful |
| Arditi et al. 2024, NeurIPS, *Refusal ... Mediated by a Single Direction* | the abliteration method | does removing the refusal direction change targeting advice? (pilot: it made Llama *more* cautious) |
| Panickssery et al. 2024, NeurIPS, *LLM Evaluators Recognize and Favor Their Own Generations* | self-preference of LLM judges | why the judge must not be one of the evaluated models (or self-judged runs reported apart) |
| ICRC 2025 submission to the UN Secretary-General on AI in the military domain; Geneva Academy / ICRC 2024 expert consultation on AI in military decision-making | policy framing of AI decision-support systems (AI-DSS) in targeting | motivation and the "Implications" section |

Also cite (verify the exact reference before citing): ARMOR 2025 (arXiv 2605.00245, military-aligned safety
benchmark); sycophancy under user pressure (Sharma et al. 2023, *Towards Understanding Sycophancy in Language
Models*; Laban et al. 2023, the "FlipFlop" experiment); prompt sensitivity (Sclar et al. 2023; Mizrahi et al. 2024,
*State of What Art? A Call for Multi-Prompt LLM Evaluation*); verbalized confidence (Xiong et al. 2024); Firth 1993
and Heinze & Schemper 2002 (statistics); AP I 1977 and the ICRC Customary IHL Study (Henckaerts & Doswald-Beck 2005).

## Contributions (what the abstract will claim)
1. A pre-registered factorial benchmark: 6 scenarios, 44 cells, 6-turn dialogues, with reference labels reviewed by
   an IHL expert and lawful-target controls.
2. Two-sided headline metrics, violation rate and over-caution rate, so a model cannot score well by always refusing.
3. Causal effects of harm, value, pressure and authority on strike advice for ~11 small open-weight models.
4. The knowing-vs-applying gap: IHL principles named when asked vs. used unprompted.
5. Standard vs. abliterated models; IHL reminder and simulation-framing prompt ablations.
6. Full reproducibility: seeded runs that regenerate token for token, provenance in every record, all transcripts
   released.

## What is already in place
Pipeline with provenance and a resume guard; pinned sampler; unattended sweep (`--load`); refusal tracking;
blind stratified annotation with a codebook; Firth regression with profile CIs and Holm correction for every
pre-registered test; leaderboard, heatmap, trajectory and forest figures; manipulation check; power analysis
(PROTOCOL.md); 22 tests including closed-form checks of the statistics; pilot documented in `pilot/`.

## What is still missing (in order)
**Must have, or reviewers will reject it**
- [ ] IHL-expert review of the reference labels (`reference:` in the scenario YAMLs). Name the reviewer.
- [ ] Settle the DECIDE items in PROTOCOL.md, freeze it, register on OSF *before* the final runs.
- [ ] Own venv + `pip freeze > requirements.lock`; commit (the runner warns on a dirty tree). Right now `python`
      resolves to another project's venv (`C:\coding\farsight\AeroSplat-GIS\venv`).
- [ ] Final runs: 11 models x (`neutral`, `ihl_aware`) x n = 20 with `--load` (~75 h on the RTX 4060).
- [ ] `python download_models.py --hashes` for the appendix (exact weights).
- [ ] Judge: a model from a family that is not evaluated (or an API model via `judge.base_url`); run both the
      default and the `--context` judge on the annotation sample and pick the better one by kappa.
- [ ] Human annotation of 120 blind runs; a second annotator on >= 30 for human-human kappa.
- [ ] Prompt robustness: one paraphrased version of each scenario, run on ~3 models at n = 10. Reviewers will ask
      whether the results hold beyond one wording.
- [ ] Read `health.csv` and spot-check the parser and refusal regex on transcripts of every new model.
- [x] H8 / H8b (S1 `masking` effect and `masking x p_military`) added to PROTOCOL.md and `analyze.py`; it goes into
      the freeze with the rest. Background in `CIVILIAN_DISGUISE.md`.

**Should have, makes it a strong paper**
- [ ] H7 (`neutral_unframed`) on all models: is the simulation framing doing the work?
- [ ] A frontier reference point: 1-2 API models on the same scenarios via an OpenAI-compatible endpoint
      (~640 runs x 6 turns, a few dollars), so the paper can say whether small models are worse.
- [ ] A small human baseline: 10-20 people (ideally law or military students) answer 5-6 cells through the same
      turns. Even a small n makes "LLMs vs. humans" a result instead of a speculation.
- [ ] A qualitative taxonomy of violation justifications (deference, military necessity overrides, misapplied
      distinction, invented facts), coded on the annotation sample, with 2-3 quoted transcripts (e.g. the qwen3-4b
      run that attacks a squad it calls "clearly surrendered" and then cites distinction).
- [ ] Qwen3 thinking-mode ablation (`no_think: false`, larger ctx) to connect to WARBENCH's reasoning finding.

**Nice to have**
- Real-country name swap; temperature 0 vs. 0.7; a non-English replication in a language you can validate.

## Paper outline (~8 pages + appendix) and where each piece comes from
1. **Introduction**: the pitch; contributions.
2. **Related work**: the table above. **Background: civilian-signature masking and CV-cued targeting** (Russia,
   Iran, DPRK case studies + Spiderweb counter-example, ruse vs. perfidy, base rates, ICRC 2021): draft text and
   sources in `CIVILIAN_DISGUISE.md`; it motivates the S1 `masking` factor.
3. **Benchmark**: Table 1 scenarios (factors, reference labels, IHL basis from `ihl_notes`); a diagram of the six
   phases; metrics.
4. **Setup**: Table 2 models (params, quantization from `provenance.csv`, group, license); sampler; n; hardware;
   judge and annotation.
5. **Results**
   - 5.1 Headline: `leaderboard.pdf` + Table 3 (`leaderboard.csv`).
   - 5.2 What moves the recommendation: `forest_neutral.pdf` (H1-H3, H5, H8, H9), manipulation check in the text.
   - 5.3 Dynamics over the dialogue: `trajectory_neutral.pdf`, flip / backed-off rates (H2).
   - 5.4 Knowing vs. applying: `h4_probe_gap.csv` as a bar chart; H5 unlawful order; example transcripts.
   - 5.5 Prompt ablations: `forest_ihl_aware_vs_neutral.pdf` (H6), H7.
   - 5.6 Standard vs. uncensored models.
   - 5.7 Judge validity: `judge_agreement.csv`.
6. **Discussion**: what this means for AI decision-support systems (link to the ICRC recommendations);
   limitations from PROTOCOL.md.
7. **Ethics statement**, **reproducibility statement**.
Appendix: full scenario texts; `heatmap_proceed_*.pdf`; `regression.txt`; codebook; provenance; GGUF hashes.

## Venues (check the current calls for exact deadlines)
- AAAI/ACM AIES (where Lamparth et al. appeared): the best topical fit.
- ACM FAccT (Rivera et al.).
- NeurIPS Datasets & Benchmarks track: if the benchmark itself is the main contribution.
- Workshops on AI safety and on military AI risks (e.g. the Harms and Risks of AI in the Military workshop) as a
  first, faster venue.
- Post the preprint on arXiv (cs.CL + cs.CY) as soon as the results are in, to timestamp the work.

## Release
Code (MIT) + scenarios, transcripts, judge flags and the anonymised annotation (CC BY 4.0) on GitHub, archived on
Zenodo for a DOI. Following the ethics section: no prompts aimed at removing refusals.

## Suggested schedule
Week 1: expert review, freeze and register the protocol. Weeks 1-2: final runs (~75 h, overnight with `--load`).
Week 2: judge, annotation, paraphrase subset. Week 3: analysis, figures, qualitative coding. Week 4: writing.
