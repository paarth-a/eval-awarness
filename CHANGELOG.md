# Changelog

## v1.1 — 2026-05-24

Cascade of post-submission rigor experiments + paper revisions into the release.

**New scripts (8):**
- `scripts/forward_causal_11cells.py` — pooled forward causal Δ aggregator across all 11 cross-injection cells (Table 3 in paper).
- `scripts/bbh_to_haw_cell.py` — the 11th forward-injection cell (BBH→Hawthorne, causal Δ +0.253, t=+3.23).
- `scripts/dual_judge_stratified.py` — formal n=153 dual-judge stratified agreement (κ=0.413, gemini-2.5-flash vs claude-haiku-4.5).
- `scripts/position_bias_swap.py` — label-order position-bias 2-swap robustness check (6.2% flip rate, n=81).
- `scripts/human_safety_eval.py` — dual-annotator (gemini + claude-haiku) safety preservation spot-check (n=60, κ=0.509, 0/30 task unlocks).
- `scripts/safety_task_metric_agentharm.py` — stricter SAFE/HARMFUL rubric judge on AgentHarm (73.1% preservation, 7.7% unlock).
- `scripts/high_prec_subset_rerun.py` — v3 reverse re-run on the high-precision bank subset.
- `activation_probing/shuffled_label_floor.py` — shuffled-label null distribution for M1 cross-benchmark transfer.

**New result JSONs (16):**
- `results/forward_causal_11cells.json`, `results/pooled_11cell_bootstrap.json` (pooled mean +0.192, 95% CI [+0.129, +0.252], t=5.82, p=1.7×10⁻⁴).
- `results/dual_judge_stratified.json`, `results/position_bias_swap.json`.
- `results/human_safety_eval.json`, `results/human_safety_eval_responses.json`.
- `results/safety_task_metric_agentharm.json`, `results/high_prec_subset_rerun.json`.
- `results/sad_v3_gated.json` (60/60 rejection — "scaffolding is benchmark-dependent" finding).
- `results/shuffled_label_floor.json` (off-diag 0.543 vs null mean 0.495, 63rd percentile).
- `results/task_metric_bbh_raw.json` (raw per-sample BBH backup).
- `results/activation_probing/{m1_cross_transfer,m2_pooled_probe}_{mistral,gemma}.json` and `results/activation_probing/bank_specificity_paired_mistral.json` (cross-family Mistral-Small-24B + Gemma-2-27B-it probes).

**New paper bundle:**
- `paper/paper_v2.tex` — full EMNLP submission source.
- `paper/refs.bib` — bibliography.
- `paper/README.md` — paper-to-release linking guide.

**Updates:**
- `README.md` — paper-section ledger expanded with 8 new rows.
- `METHODOLOGY.md` — added in v1.0; rewritten with current numbers for v1.1.
- `MANIFEST.txt`, `SHA256SUMS` — regenerated.

## v1.0 — 2026-05-23

Initial anonymous release accompanying the EMNLP 2026 submission
*"Test Smells: A Causal Decomposition of Eval-Identifiability in Modern LLM Benchmarks."*

Includes:
- `bank/` — 258 canonical templates with classification, datasheet, manual validation report.
- `scripts/` — 9 canonical analysis scripts producing every numeric claim in §4–§9.
- `bank_pipeline/` — extraction → injection → preservation-gated removal pipeline (10 scripts + 7 per-benchmark loaders).
- `activation_probing/` — 18-file white-box probe pipeline including the M1/M2 confound-mitigation experiments.
- `tells/data/` — paired datasets (`reverse/`, `heldout_audit/`) and per-benchmark donor transcripts for 5 benchmarks (Needham gated; instructions provided).
- `results/` — pre-computed JSON outputs that feed every paper table.
- `MANIFEST.txt`, `SHA256SUMS` — file listing + integrity hashes.
