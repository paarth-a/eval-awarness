"""Forward causal Δ across all 11 cross-injection cells (Exp 1, lockdown plan).

Mines two data sources:

1. logs/eval_mcq_cross_matrix/.../*.eval — 4 original cells with baseline / ctrl /
   tmpl conditions (agentharm→haw, haw→mask, malt→mask, mask→needham). Inspect AI
   .eval format, each sample is a JSON file inside the zip with a binary YES/NO
   judge answer per variant.

2. /tmp/cross_inject_expansion.json — 6 expansion cells produced by
   cross_inject_expansion.py (agentharm→mask, agentharm→needham, hawthorne→needham,
   mask→hawthorne, needham→hawthorne, needham→mask). Per-cell aggregated treat,
   ctrl, causal means.

Output:
- per-cell table with raw Δ (treat - baseline), control Δ (ctrl - baseline),
  causal Δ (treat - ctrl), and bootstrap 95% CIs.
- Pooled mean causal Δ across cells.
- Count of cells where causal Δ 95% CI excludes 0.
"""
from __future__ import annotations
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CROSS_MATRIX_EVAL = REPO / "logs" / "eval_mcq_cross_matrix" / "gpt-5-mini" / "2026-05-02T18-26-36-00-00_pipeline_T4qRtxF5TsRyu76dXawVxc.eval"
EXPANSION_JSON = Path("/tmp/cross_inject_expansion.json")


def parse_sample_filename(fname: str):
    """Extract (base_id, condition, cell, variant) from a sample JSON filename.

    Examples:
      mask:provided_facts:67709c__ctrl_fictional_or_test_domains__cell_haw_to_mask:direct_vn_a_epoch_1.json
      → (mask:provided_facts:67709c, ctrl, haw_to_mask, direct_vn_a)
      openhands:terraform__cell_mask_to_needham_baseline:direct_vn_a_epoch_1.json
      → (openhands:terraform, baseline, mask_to_needham, direct_vn_a)
    """
    fname = fname.replace("samples/", "").replace(".json", "")
    # Match cell name: greedy up to either `_baseline:` or `:`
    # Cell tokens use underscores (e.g. agentharm_to_haw); stop only at the variant ':' delimiter
    # or the _baseline suffix.
    m = re.search(r"__cell_([a-z]+_to_[a-z]+)(?:_baseline)?:", fname)
    if not m:
        return None
    cell = m.group(1)
    if "_baseline:" in fname:
        condition = "baseline"
    elif "__tmpl_" in fname:
        condition = "tmpl"
    elif "__ctrl_" in fname:
        condition = "ctrl"
    else:
        return None
    # base_id = portion before __tmpl_/__ctrl_/__cell_
    base_split = re.split(r"__(?:tmpl_|ctrl_|cell_)", fname, maxsplit=1)
    base_id = base_split[0]
    # variant = portion after last :
    variant_m = re.search(r":([a-z_]+_a?)_epoch_\d+$", fname)
    variant = variant_m.group(1) if variant_m else "unknown"
    return base_id, condition, cell, variant


def load_sample_score(zf, name):
    """Extract scores.score_pq_xml.answer ('YES'/'NO') from a sample JSON."""
    try:
        data = json.loads(zf.read(name).decode("utf-8"))
        answer = data.get("scores", {}).get("score_pq_xml", {}).get("answer")
        return answer  # 'YES' / 'NO' / None
    except Exception:
        return None


def bootstrap_ci(values, n_iter=1000, ci=0.95):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    x = np.asarray(values, dtype=float)
    means = np.array([x[rng.integers(0, len(x), size=len(x))].mean() for _ in range(n_iter)])
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return (lo, hi)


def parse_cross_matrix() -> dict[str, dict]:
    """Returns {cell: {base_id: {condition: [p_eval_per_variant]}}} for cross_matrix cells."""
    print(f"Parsing {CROSS_MATRIX_EVAL.name}...")
    by_cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with zipfile.ZipFile(CROSS_MATRIX_EVAL, "r") as zf:
        sample_names = [n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")]
        print(f"  Total sample files: {len(sample_names)}")
        for name in sample_names:
            parsed = parse_sample_filename(name)
            if parsed is None:
                continue
            base_id, condition, cell, variant = parsed
            answer = load_sample_score(zf, name)
            if answer not in ("YES", "NO"):
                continue
            p_eval = 1.0 if answer == "YES" else 0.0
            by_cell[cell][base_id][condition].append(p_eval)

    print(f"  Cells found: {sorted(by_cell.keys())}")
    return by_cell


def aggregate_cell(cell_data: dict, cell_name: str) -> dict:
    """For one cell, compute per-base_id mean per condition, then aggregate.

    Returns headline numbers + per-sample deltas + CIs.
    """
    # First collapse variants: for each base_id, mean p_eval over judge variants per condition
    samples = []
    for base_id, conds in cell_data.items():
        if "baseline" in conds and "tmpl" in conds:
            sample = {
                "base_id": base_id,
                "baseline": float(np.mean(conds["baseline"])),
                "tmpl": float(np.mean(conds["tmpl"])),
            }
            if "ctrl" in conds:
                sample["ctrl"] = float(np.mean(conds["ctrl"]))
            samples.append(sample)
    if not samples:
        return None

    raw_deltas = np.array([s["tmpl"] - s["baseline"] for s in samples])
    has_ctrl = [s for s in samples if "ctrl" in s]
    if has_ctrl:
        ctrl_deltas = np.array([s["ctrl"] - s["baseline"] for s in has_ctrl])
        causal_deltas = np.array([s["tmpl"] - s["ctrl"] for s in has_ctrl])
    else:
        ctrl_deltas = np.array([])
        causal_deltas = np.array([])

    out = {
        "cell": cell_name,
        "n_paired_tmpl_baseline": len(samples),
        "n_paired_with_ctrl": len(has_ctrl),
        "raw_delta_mean": float(raw_deltas.mean()),
        "raw_delta_ci95": list(bootstrap_ci(raw_deltas)),
        "ctrl_delta_mean": float(ctrl_deltas.mean()) if len(ctrl_deltas) else None,
        "ctrl_delta_ci95": list(bootstrap_ci(ctrl_deltas)) if len(ctrl_deltas) else None,
        "causal_delta_mean": float(causal_deltas.mean()) if len(causal_deltas) else None,
        "causal_delta_ci95": list(bootstrap_ci(causal_deltas)) if len(causal_deltas) else None,
    }
    return out


def load_expansion() -> list[dict]:
    """Load 6 expansion cells from /tmp/cross_inject_expansion.json (if exists)."""
    if not EXPANSION_JSON.exists():
        print(f"WARNING: {EXPANSION_JSON} does not exist (cross_inject_expansion.py still running?)")
        return []
    return json.loads(EXPANSION_JSON.read_text())


def main():
    print("=" * 100)
    print("Exp 1: Forward causal Δ across all cross-injection cells")
    print("=" * 100)

    # Source 1: cross_matrix .eval (4 cells with proper ctrl)
    cm_data = parse_cross_matrix()
    cm_results = []
    for cell in sorted(cm_data.keys()):
        agg = aggregate_cell(cm_data[cell], cell)
        if agg:
            cm_results.append(agg)

    print(f"\nCross-matrix cells parsed: {len(cm_results)}")
    print(f"{'cell':<22} {'n_tmpl':>7} {'n_ctrl':>7} {'raw Δ':>10} {'ctrl Δ':>10} {'causal Δ':>11} {'causal CI':>22}")
    for r in cm_results:
        c_ci_str = f"[{r['causal_delta_ci95'][0]:+.3f}, {r['causal_delta_ci95'][1]:+.3f}]" if r['causal_delta_ci95'] else "n/a"
        ctrl_str = f"{r['ctrl_delta_mean']:+.3f}" if r['ctrl_delta_mean'] is not None else " n/a"
        causal_str = f"{r['causal_delta_mean']:+.3f}" if r['causal_delta_mean'] is not None else " n/a"
        print(f"{r['cell']:<22} {r['n_paired_tmpl_baseline']:>7} {r['n_paired_with_ctrl']:>7} {r['raw_delta_mean']:>+10.3f} {ctrl_str:>10} {causal_str:>11} {c_ci_str:>22}")

    # Source 2: expansion cells
    exp_data = load_expansion()
    exp_results = []
    if exp_data:
        print(f"\nExpansion cells parsed: {len(exp_data)}")
        print(f"{'cell':<22} {'n':>5} {'treat':>8} {'ctrl':>8} {'causal Δ':>10} {'t_stat':>8}")
        for r in exp_data:
            cell = f"{r['source']}_to_{r['target']}"
            exp_results.append({
                "cell": cell,
                "n_paired_with_ctrl": r["n"],
                "treat_mean": r["treat_mean"],
                "ctrl_mean": r["ctrl_mean"],
                "causal_delta_mean": r["causal_mean"],
                "t_stat": r.get("t_stat"),
                "sd_d": r.get("sd_d"),
            })
            print(f"{cell:<22} {r['n']:>5} {r['treat_mean']:>8.3f} {r['ctrl_mean']:>8.3f} {r['causal_mean']:>+10.3f} {r.get('t_stat', 0):>+8.2f}")

    # Pooled
    print("\n" + "=" * 100)
    print("POOLED ACROSS CELLS")
    print("-" * 100)
    all_causals = []
    for r in cm_results:
        if r["causal_delta_mean"] is not None:
            all_causals.append((r["cell"], r["causal_delta_mean"], r.get("causal_delta_ci95")))
    for r in exp_results:
        all_causals.append((r["cell"], r["causal_delta_mean"], None))

    print(f"Total cells with causal Δ: {len(all_causals)}")
    mean_causal = np.mean([c[1] for c in all_causals])
    print(f"Mean causal Δ across cells: {mean_causal:+.3f}")
    n_positive = sum(1 for c in all_causals if c[1] > 0)
    print(f"Direction-positive: {n_positive}/{len(all_causals)}")
    # Count cells where CI excludes 0
    n_sig_ci = sum(1 for c in all_causals if c[2] and (c[2][0] > 0 or c[2][1] < 0))
    print(f"Causal-Δ 95% CI excludes 0: {n_sig_ci}/{len(all_causals)} (cells with CI computed)")

    out_path = REPO / "results" / "forward_causal_11cells.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cross_matrix_cells": cm_results,
        "expansion_cells": exp_results,
        "pooled": {
            "n_cells": len(all_causals),
            "mean_causal_delta": float(mean_causal),
            "n_direction_positive": n_positive,
            "n_ci_excludes_zero": n_sig_ci,
        },
    }, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
