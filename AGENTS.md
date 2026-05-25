# AGENTS.md

Anonymous code-and-data release for the EMNLP 2026 submission *"Test Smells: A Causal Decomposition of Eval-Identifiability in Modern LLM Benchmarks."* This repository ships pre-computed result JSONs alongside the scripts that produced them; the primary AI-agent task is reproducing / extending paper claims, **not** building new product features.

## Orientation (read these first)

- [README.md](README.md) — quickstart, full paper-section → script → output ledger, end-to-end reproduction recipe. **The ledger is the source of truth** for "which script produces which paper number."
- [METHODOLOGY.md](METHODOLOGY.md) — methodological choices and rationale.
- [activation_probing/README.md](activation_probing/README.md) — probe pipeline internals (extraction → train → apply → confound mitigations).
- [bank/README.md](bank/README.md), [bank/DATASHEET.md](bank/DATASHEET.md) — the 258-template bank artifact.
- [paper/README.md](paper/README.md) — LaTeX source and paper-claim ↔ artifact mapping.

## Repository shape

- `bank/` — the 258-template bank (JSON artifacts). Read-only artifact.
- `bank_pipeline/` — extraction → classification → forward injection (`merge_tells.py`) → reverse removal (`remove_tells.py`) → controls. `loaders/` holds per-benchmark donor adapters (agentharm, aisi, bbh, hawthorne, malt, mask, sad).
- `scripts/` — paper-section-aligned analysis scripts. Filename ≈ paper claim (e.g. `forward_causal_11cells.py`, `dual_judge_stratified.py`, `position_bias_swap.py`).
- `activation_probing/` — standalone white-box probe pipeline (§8). Run as a module: `python -m activation_probing.<name>`. Driven by `run_full_pipeline.sh` + env vars.
- `tells/data/` — paired datasets. Per-benchmark subdirs hold donor + `forward_paired/` + `reverse_gated/`. `reverse/` and `heldout_audit/` hold cross-benchmark gated-reverse outputs.
- `results/` — shipped reference JSONs. Re-runs should diff against these.
- `paper/` — LaTeX source (`paper_v2.tex`, `refs.bib`).

## Conventions

- **Paths**: always resolve relative to `Path(__file__).resolve().parent.parent` (the release root). Never hardcode absolute paths — the release was audited to remove `/Users/`, `/home/`, `/workspace/` prefixes.
- **Secrets**: `OPENROUTER_API_KEY` (judge/rewriter calls) and `HF_TOKEN` (gated Needham donor) live in `.env`. See [.env.example](.env.example). Never commit `.env`; never inline keys.
- **Probe model selection**: controlled by `ACTPROBE_MODEL` env var (e.g. `Qwen/Qwen3.5-27B`, `mistralai/Mistral-Small-24B-Instruct-2501`, `unsloth/gemma-2-27b-it`). `ACTPROBE_N_PER` caps samples per cell. Probe outputs are suffixed `__<model>.json`.
- **Result naming**: `results/<script_basename>.json`; probe outputs go in `results/activation_probing/`. When adding a script, mirror this and update the README ledger.
- **Determinism**: scripts ship pre-computed JSONs in `results/`. A change that alters those numbers should be intentional and noted in `CHANGELOG.md`.

## Common commands

```bash
# Smoke test — no API, no GPU, deterministic:
python scripts/lexical_baseline.py          # → results/lexical_baseline_lobo.json

# Re-run an analysis script (CPU, hits OpenRouter for judge calls):
python scripts/forward_causal_11cells.py    # → results/forward_causal_11cells.json

# Probe pipeline (GPU required; ~$5/H200-hour):
export ACTPROBE_MODEL=Qwen/Qwen3.5-27B
bash activation_probing/run_full_pipeline.sh

# Probe sub-step as a module (always `-m`, never `python activation_probing/foo.py`):
python -m activation_probing.m1_cross_transfer
```

There is no test suite, no linter config, and no build step. Validation = "re-run the script and diff the output JSON against `results/`."

## Pitfalls

- **Don't `python activation_probing/x.py` directly** — the package uses relative imports; always use `python -m activation_probing.x`.
- **Gated datasets**: the Needham donor requires HF access approval before `HF_TOKEN` works. `HF_TELLS_REPO` can point at a local clone to bypass.
- **`HF_HOME` disk**: larger Qwen models need ≥200 GB free; set `HF_HOME` to a roomy volume.
- **Anonymity**: this is a double-blind submission release. Do **not** add author names, original repo URLs, internal hostnames, or absolute paths from a dev machine. Re-audit anything you generate.
- **Two bank variants exist**: `templates_classified.json` (canonical) vs `templates_classified_aggressive.json` (alt threshold). Don't silently swap them.
- **License**: CC-BY-NC-SA-4.0. Upstream benchmark transcripts in `tells/data/` are redistributed under their own licenses — don't relicense.

## When extending

- New analysis → add to `scripts/`, write JSON to `results/`, append a row to the README paper-section ledger.
- New benchmark loader → add to `bank_pipeline/loaders/` following the existing loader signatures; register in `bank_pipeline/data_config.py`.
- New probe confound mitigation → add to `activation_probing/`, expose via `run_full_pipeline.sh` if it should run by default.
