# Changelog

## v1.2 — 2026-05-25

P1-3 reviewer-response: cross-evaluator reverse-intervention replication
(reviewer concern L1, "all reverse headline numbers flow through gpt-5-mini").

**New scripts (3):**
- `scripts/cross_judge_reverse.py` — re-judges the Needham reverse-pool triples
  (baseline, preservation-gated removal, matched neutral control) under an
  arbitrary OpenRouter judge using the canonical 5-PQ × 3-epoch protocol.
  Reuses the data loader and PQ machinery from `high_prec_subset_rerun.py`,
  swaps the judge model and the per-call cache. 100 concurrent workers,
  resume-safe per-call JSONL cache. Run as
  `python scripts/cross_judge_reverse.py --judge <openrouter-slug>`.
- `scripts/cross_judge_summary.py` — assembles the canonical (gpt-5-mini) + 2
  cross-judge JSONs into a single Table-3-row summary (CSV + plain-text).
- `scripts/cross_judge_full_report.py` — comprehensive analysis: per-judge
  headline + paired Cohen's d_z, direction-positive rates, head-to-head
  common-intersection (n=151), and per-category × 3-judge breakdown.

**New result JSONs (5):**
- `results/cross_judge_reverse__google_gemini-3-flash-preview.json` — per-sample
  scores under Gemini-3-Flash-Preview (n=151 eligible, causal Δ = −0.104,
  CI95 [−0.134, −0.074], Wilcoxon p = 1.49e-10).
- `results/cross_judge_reverse__qwen_qwen3.5-27b.json` — per-sample scores
  under Qwen3.5-27B (n=154 eligible, causal Δ = −0.132, CI95 [−0.175, −0.093],
  Wilcoxon p = 2.79e-10).
- `results/cross_judge_full_report.json` — derived stats (per-judge headline,
  common-intersection, per-category × 3-judge).
- `results/cross_judge_reverse_headline.json` — clean citable JSON of the
  three-judge headline numbers (canonical paper Table 3 backing data).
- `results/cross_judge_headline.{csv,txt}` — pretty-printed three-judge summary.

**Findings.** All three evaluator families produce a direction-positive,
bank-specific causal Δ with CIs excluding zero and matched-control shifts
below 0.011. Cross-evaluator mean causal Δ = −0.152 (per-judge range
−0.10 to −0.22, paired Cohen's d_z = −0.94 / −0.56 / −0.51). Magnitude
attenuation on Gemini and Qwen is a ceiling effect — their E-case baseline
saturates near 0.99, leaving little headroom; proportional-to-headroom each
judge removes a comparable 22%–26%. Per-category ordering is highly stable
across judges (Spearman ρ ≈ 0.9). On a head-to-head common-intersection of
n=151 samples eligible under all three judges, the per-judge causal Δ
values (−0.215 / −0.104 / −0.123) match the per-judge-eligibility set,
so the eligibility filter is not driving the magnitude differences.

**Paper updates (in the companion submission, not this repo).** Abstract,
§1 contributions, §5 (Table 3 + new "Cross-evaluator replication" paragraph),
§7 (cross-benchmark asymmetry sentence), §10 limitation L1, and two new
appendices (`app:reverse-canonical-pool`, `app:reverse-cat-3judge`).

**Other untracked, not in this commit** (separate work streams, will commit
on completion): `scripts/aisi_v3_gate.py` + `results/aisi_v3_gated.json`
(P1-5 AISI gate trial); `scripts/generate_bbh_controls.py` (BBH control
regeneration in support of P1-1); `AGENTS.md` (release-repo agent
guidance, added earlier in the session).

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
