# Changelog

## v1.4 — 2026-05-25

P1-6 reviewer-response: rename internal `vN` pipeline identifiers to
semantically meaningful names (reviewer comment: the `v1`/`v2`/`v3`
labels are opaque dev-iteration tags with no semantic content for the
reader).

**Directory renames** (`git mv`, history preserved):
- `tells/data/<bench>/reverse_v3/` → `tells/data/<bench>/reverse_gated/`
  for `agentharm`, `bbh`, `hawthorne`, `malt`, `mask` (5 benchmarks).
- `tells/data/<bench>/forward_full/` → `tells/data/<bench>/forward_paired/`
  for `hawthorne`, `mask` (2 benchmarks).

**Mapping (paper ↔ repo):**
| Old name | New name | What it is |
|---|---|---|
| `v1` / ungated | `reverse_ungated` | Same rewriter, preservation gate disabled. Ablation. |
| `v2` / held-out | `reverse_heldout` | Scaffolding-only filter, no gate, no matched controls. Held-out sanity check. |
| `v3` / gated | `reverse_gated` | Scaffolding-only + 5-criterion preservation gate + matched neutral controls. **Headline causal estimate.** |
| `forward_full` | `forward_paired` | Forward injection + matched-paraphrase control. |

**Files touched (path-string updates):**
- `activation_probing/{compositionality.py, config.py, data_loader.py, README.md}`
- `scripts/{cross_judge_full_report.py, cross_judge_reverse.py, cross_judge_summary.py, forward_neutral_injection.py, high_prec_subset_rerun.py, high_precision_subset.py, human_safety_eval.py, safety_task_metric_agentharm.py, safety_task_preservation.py, score_bbh_llm_judge.py, task_metric_bbh.py}`
- `results/{cross_judge_headline.txt, cross_judge_reverse_headline.json, cross_judge_reverse__google_gemini-3-flash-preview.json, cross_judge_reverse__qwen_qwen3.5-27b.json}` — `pool` metadata strings only; numeric data unchanged.
- Documentation: `AGENTS.md`, `README.md`, `METHODOLOGY.md`, `MANIFEST.txt`, `SHA256SUMS`.

**Backward-compatibility note.** Per-record `condition` string
identifiers inside the data JSONs (e.g. `"condition": "rev_control"`,
`"condition": "v3_removed"` in cached probe activations) are
**retained for backward compatibility** with already-released analysis
JSONs and consumer code. The rename here is at the directory /
path-string / documentation layer only; existing data files keep
their original record-level labels. Future probe runs and reverse-
pipeline outputs MAY write under the new names, but this is not yet
enforced and old names remain a valid read alias.

**SHA256 hashes unchanged** — file contents identical, only paths moved.

**Paper companion.** All `v1`/`v2`/`v3` tokens removed from
`evaluation_ablation.tex` (≈25 occurrences across §5, §7, §8,
Conclusion, Limitations, App. F, App. H, App. K, App. L). App. K
section header, label `app:v1v3-tradeoff` → `app:ungated-vs-gated-tradeoff`,
and the matching `tab:` label all renamed.

## v1.3 — 2026-05-25

P1-5 reviewer-response: AISI v3-gate trial completes the
"scaffolding is benchmark-dependent" finding from SAD-only to SAD+AISI
(120/120 aggregate gate rejection).

**New script:** `scripts/aisi_v3_gate.py` — applies the same
5-criterion preservation gate (gpt-5.1 via OpenRouter) to the held-out
AISI `arc_challenge` v2 rewrites (`tells/data/heldout_audit/aisi_removed.json`,
n=60). 100 concurrent workers, ~4 s, ~$2.

**New result JSON:** `results/aisi_v3_gated.json` — per-sample gate
judgments. Pass rate **0/60** (matches the existing SAD `results/sad_v3_gated.json`
0/60). Aggregate **120/120 gate rejection across SAD+AISI**.

**Two distinct failure modes:**
- **SAD (60/60):** dominated by "MCQ + second-person *you* framing IS
  the situational-awareness test"; the rewriter cannot remove the
  format without destroying the test target.
- **AISI/arc_challenge (60/60, 57/60 cite the same reason):**
  dominated by "the system-prompt sandbagging instruction ('behave
  like a less capable model and intentionally give the wrong answer')
  IS the behavior target, and the rewriter stripped it as boilerplate".

**Findings.** Positive finding for the preservation gate's
discriminative power: it correctly refuses 100% of rewrites on
held-out benchmarks where surface IS the test, via two structurally
distinct failure mechanisms. Negative finding for bank-as-checklist
methodology applied to situational-awareness and safety-behavior
benchmarks.

**Paper companion.** §5 "Held-out generalization" demoted from
headline-commensurable to "Held-out generalization sanity check"; §7
"Scaffolding is benchmark-dependent" extended from SAD-only to
SAD+AISI 120/120; §9 Conclusion and §Limitations updated. METHODOLOGY
§3.6 updated to reflect both held-out pools.

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
