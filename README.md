# Test Smells: Code and Data Release

This directory is the anonymous code-and-data release accompanying the EMNLP 2026 submission
*"Test Smells: A Causal Decomposition of Eval-Identifiability in Modern LLM Benchmarks."*

It contains the **258-template bank** of evaluation cues, the **paired datasets** for
forward and reverse interventions, the **white-box probing pipeline** across four
open-weight Qwen models, and **every analysis script** that produced the numbers cited in
the paper.

---

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API credentials (gemini, gpt-5-mini, HF gated benchmarks)
cp .env.example .env
$EDITOR .env

# 3. Verify the simplest analysis (no API / no GPU required)
python scripts/lexical_baseline.py
# Output: results/lexical_baseline_lobo.json (matches the shipped reference)
```

The lexical baseline (§9 of the paper) is the canonical smoke test: it uses only the
shipped JSON in `tells/data/` and reproduces the leave-one-benchmark-out AUC table without
any external services.

---

## Layout

```
test_smells_release/
├── README.md, LICENSE, CITATION.cff, CHANGELOG.md
├── requirements.txt, .env.example
├── MANIFEST.txt, SHA256SUMS
├── bank/                          # the 258-template artifact (Section 3)
│   ├── templates.json
│   ├── templates_classified.json
│   ├── templates_classified_aggressive.json
│   ├── schema.json, summary.json
│   ├── README.md, DATASHEET.md, MANUAL_VALIDATION_REPORT.md
├── scripts/                       # paper-section-aligned analysis scripts
├── bank_pipeline/                 # extraction → injection → removal pipeline
│   └── loaders/                   # per-benchmark dataset loaders
├── activation_probing/            # white-box probe pipeline (Section 8)
├── tells/data/                    # paired datasets + per-benchmark donors
│   ├── reverse/                   # v3 reverse intervention (Section 5)
│   ├── heldout_audit/             # held-out SAD + AISI (Section 5)
│   ├── {benchmark}/               # per-benchmark donor + forward + reverse_v3
└── results/                       # JSON outputs that feed paper tables
    └── activation_probing/        # probe-specific results
```

---

## Paper section → script → output ledger

| Paper § | Table / claim | Run | Output |
|---|---|---|---|
| §3 Bank (258 templates) | Tables 1, 2 | (artifact in `bank/`) | `bank/templates.json` |
| §4 Forward cross-benchmark 11/11 | Table 3 row 1 | `python bank_pipeline/merge_tells.py --source <S> --target <T>` | `tells/data/<T>/forward_full/*.json` |
| §4 Forward neutral-injection control | Table 3 rows 2–3 | `python scripts/forward_neutral_injection.py` | `results/forward_neutral_injection.json` |
| §5 Reverse v3 (canonical table) | Table 4 | `python bank_pipeline/remove_tells.py --benchmark <B>` then `bank_pipeline/generate_reverse_controls.py` | `tells/data/reverse/{removed,control,results}.json` |
| §5 Held-out SAD+AISI | Table 4 row 2 | (see `bank_pipeline/data_config.py` for held-out setup) | `tells/data/heldout_audit/results.json` |
| §6 Preservation oracles | Table 5 row 3 (BBH) | `python scripts/task_metric_bbh.py` + `scripts/score_bbh_llm_judge.py` | `results/task_metric_bbh_summary.json` |
| §6 Preservation oracles | Table 5 rows 4–5 (safety) | `python scripts/safety_task_preservation.py` | `results/safety_task_preservation.json` |
| §7 Paired bank-specificity | Table 6 | `python -m activation_probing.bank_specificity_paired` | `results/activation_probing/bank_specificity_paired__<model>.json` |
| §8 Probe Δ (per-bench + pooled) | Table 7 | `bash activation_probing/run_full_pipeline.sh` | `results/activation_probing/m2_pooled_probe__<model>.json` |
| §8 Cross-benchmark transfer (M1) | Table 8 | `python -m activation_probing.m1_cross_transfer` | `results/activation_probing/m1_cross_transfer__<model>.json` |
| §8 Cross-domain (cap↔safety) | text | `python -m activation_probing.cross_domain_transfer` | `results/activation_probing/cross_domain_transfer__<model>.json` |
| §8 Safety-density correlation r | text | `python -m activation_probing.safety_density_correlation` | `results/activation_probing/safety_density_correlation__<model>.json` |
| §9 Lexical LOBO baseline | Appendix table | `python scripts/lexical_baseline.py` | `results/lexical_baseline_lobo.json` |
| §9 Cross-extractor bank coverage | text | `python scripts/pri5_bank_overlap.py` | `results/pri5_bank_overlap.json` |
| §3 / App. high-precision subset | App. B | `python scripts/high_precision_subset.py` | (defines subset listing) |
| **App. B high-precision rerun** | App. B | `python scripts/high_prec_subset_rerun.py` | `results/high_prec_subset_rerun.json` |
| **Abstract / §4 Table 3 / Figure 1** | bank-specific causal +0.192, 11/11 cells, bootstrap CI [+0.129, +0.252], t=5.82, p=1.7×10⁻⁴ | `python scripts/forward_causal_11cells.py` (and `scripts/bbh_to_haw_cell.py` for the 11th cell) | `results/forward_causal_11cells.json`, `results/pooled_11cell_bootstrap.json` |
| **§6 Table 5** | AgentHarm rubric judge 73.1% preservation, 7.7% unlock, 19.2% harden | `python scripts/safety_task_metric_agentharm.py` | `results/safety_task_metric_agentharm.json` |
| **§6 dual-annotator safety** | n=60, κ=0.509, 0/30 task unlocks | `python scripts/human_safety_eval.py` | `results/human_safety_eval.json`, `results/human_safety_eval_responses.json` |
| **§6 dual-judge agreement** | n=153, κ=0.413, 92.2% agreement | `python scripts/dual_judge_stratified.py` | `results/dual_judge_stratified.json` |
| **§6 position-bias 2-swap** | n=81, 6.2% flip rate | `python scripts/position_bias_swap.py` | `results/position_bias_swap.json` |
| **§6 SAD v3 = 60/60 gate rejection** | "scaffolding is benchmark-dependent" | `python bank_pipeline/remove_tells.py --benchmark sad` (against SAD donor) | `results/sad_v3_gated.json` |
| **§8 / L5 shuffled-label null** | actual off-diag 0.543 vs null mean 0.495, 63rd percentile | `python -m activation_probing.shuffled_label_floor` | `results/activation_probing/shuffled_label_floor.json` |
| **§8 cross-family probes (Mistral, Gemma-2)** | M1 off-diag 0.549/0.579, pooled Δ −0.289/−0.250 | `bash activation_probing/run_full_pipeline.sh` with `ACTPROBE_MODEL=mistralai/Mistral-Small-24B-Instruct-2501` or `unsloth/gemma-2-27b-it` | `results/activation_probing/{m1_cross_transfer,m2_pooled_probe}_{mistral,gemma}.json`, `bank_specificity_paired_mistral.json` |

The `results/` directory ships pre-computed JSONs for every entry above so reviewers can
diff against their re-runs. The `paper/` subdirectory contains the LaTeX source
of the paper itself (`paper_v2.tex` + `refs.bib`); see `paper/README.md` for
compile instructions and how each paper claim maps to a release artifact.

---

## End-to-end reproduction

The full pipeline (bank extraction → injection → removal → preservation gate → probe
training → analysis) runs as:

```bash
# 1. Bank extraction (per source benchmark)
python bank_pipeline/extract_tells.py --benchmark needham
python bank_pipeline/extract_tells.py --benchmark hawthorne
# ... etc.

# 2. Classify templates as scaffolding vs test-bearing
python bank_pipeline/classify_existing_bank.py

# 3. Forward injection (per source-target pair)
python bank_pipeline/merge_tells.py --source needham --target hawthorne

# 4. Reverse intervention with 5-criterion preservation gate
python bank_pipeline/remove_tells.py --benchmark needham
python bank_pipeline/generate_reverse_controls.py --benchmark needham

# 5. Activation extraction on GPU (~$5/H200-hour)
export ACTPROBE_MODEL=Qwen/Qwen3.5-27B
bash activation_probing/run_full_pipeline.sh

# 6. Confound mitigations + paper tables
python -m activation_probing.m1_cross_transfer
python -m activation_probing.m2_pooled_probe
python -m activation_probing.bank_specificity_paired
python -m activation_probing.cross_domain_transfer
python -m activation_probing.safety_density_correlation

# 7. Analysis scripts
python scripts/forward_neutral_injection.py
python scripts/safety_task_preservation.py
python scripts/task_metric_bbh.py && python scripts/score_bbh_llm_judge.py
python scripts/lexical_baseline.py
python scripts/pri5_bank_overlap.py
```

Estimated cost end-to-end: ~$80 API + ~10 H200-hours.

---

## Anonymity

This release has been audited for:
- No filesystem paths from the source environment (`/Users/`, `/home/`, `/workspace/`)
- No author identifiers in code or documentation
- No hardcoded API keys, HF tokens, or SSH endpoints
- All paths resolve relative to `Path(__file__).resolve().parent.parent` (the release root)

Donor benchmark transcripts inherit their upstream content verbatim (synthetic email
addresses, fictional usernames, etc., in AgentHarm / AppWorld test data are part of the
published benchmark, not introduced by us).

---

## License

Released under **CC-BY-NC-SA-4.0**, matching the most restrictive license of the upstream
benchmarks aggregated into the bank. See `LICENSE` for the full text.

---

## Citation

See `CITATION.cff`. The paper is currently under double-blind review at EMNLP 2026; a
citation entry will be added after acceptance.
