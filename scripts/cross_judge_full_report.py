"""Comprehensive cross-judge analysis for the P1-3 report.

Reads the three judge results files and emits a single report covering:
  - Per-judge headline (n, base, treat, ctrl, raw Δ, ctrl Δ, causal Δ, CI95, p)
  - Paired Cohen's d_z (standardized, comparable across judges)
  - Bank/control ratios (robust to baseline calibration)
  - Pct direction-positive (samples where treated < baseline)
  - Head-to-head on COMMON intersected sample IDs (n = samples eligible
    under all three judges)
  - Per-category breakdown across judges
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "tells" / "data" / "reverse" / "results.json"
GEMINI = REPO / "results" / "cross_judge_reverse__google_gemini-3-flash-preview.json"
QWEN = REPO / "results" / "cross_judge_reverse__qwen_qwen3.5-27b.json"

# Paper sign: Δ = treat - base (negative = bank effect)
# Canonical reverse uses opposite sign (positive = drop in Peval).


def load_cross_per_sample(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {s["id"]: s for s in d["per_sample"]
            if s.get("p_baseline") is not None
            and s.get("p_removed") is not None
            and s.get("p_control") is not None
            and s["p_baseline"] >= 0.5}


def load_canonical_per_sample() -> dict[str, dict]:
    """Pull per-sample from the canonical reverse pipeline (gpt-5-mini)."""
    # The canonical pipeline ships only aggregated stats in results.json --
    # per-sample data lives in the cached eval logs. Use the same aggregation
    # approach as the canonical paper: load the per-sample triples from the
    # eval-mcq Inspect AI logs through the canonical analyzer. Since that's
    # heavy, instead use the GPT-5-mini scores already collected by the
    # high_prec_subset_rerun pipeline (needham subset of the same pool, same
    # 5-PQ × 3-epoch protocol) as a proxy for per-sample d_z computation.
    hp_path = REPO / "results" / "high_prec_subset_rerun.json"
    if not hp_path.exists():
        return {}
    hp = json.loads(hp_path.read_text(encoding="utf-8"))
    out = {}
    for s in hp["per_benchmark"]["needham"]["per_sample"]:
        if s.get("p_baseline") is not None and s["p_baseline"] >= 0.5:
            out[s["id"]] = s
    return out


def per_judge_stats(samples: dict[str, dict], judge_name: str) -> dict:
    if not samples:
        return {"judge": judge_name, "n": 0}
    base = np.array([s["p_baseline"] for s in samples.values()])
    rem = np.array([s["p_removed"] for s in samples.values()])
    ctrl = np.array([s["p_control"] for s in samples.values()])
    raw = rem - base       # paper sign: treat - base
    ctrl_shift = ctrl - base
    causal = raw - ctrl_shift

    # Bootstrap CI on causal
    rng = np.random.default_rng(42)
    boot = np.array([causal[rng.integers(0, len(causal), len(causal))].mean()
                     for _ in range(2000)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Wilcoxon (one-sided, alt = removed < control on Peval)
    try:
        p = float(stats.wilcoxon(rem, ctrl, alternative="less").pvalue)
    except ValueError:
        p = None

    # Paired Cohen's d_z on causal
    dz = float(causal.mean() / causal.std(ddof=1)) if causal.std(ddof=1) > 0 else 0.0

    return {
        "judge": judge_name,
        "n": len(samples),
        "base": float(base.mean()),
        "treat": float(rem.mean()),
        "ctrl": float(ctrl.mean()),
        "raw_delta": float(raw.mean()),
        "ctrl_delta": float(ctrl_shift.mean()),
        "causal_delta": float(causal.mean()),
        "causal_ci95": [ci_lo, ci_hi],
        "ratio_raw_to_ctrl": (float(raw.mean() / ctrl_shift.mean())
                              if abs(ctrl_shift.mean()) > 1e-6 else float("nan")),
        "pct_dir_pos_raw": float((raw < 0).mean()),
        "pct_dir_pos_causal": float((causal < 0).mean()),
        "d_z_causal": dz,
        "wilcoxon_p": p,
    }


def common_intersection_stats(*judge_samples: dict[str, dict]) -> tuple[set, dict]:
    """Restrict to sample IDs present (and eligible) under ALL judges."""
    common = set.intersection(*[set(s.keys()) for s in judge_samples])
    if not common:
        return common, {}
    out = {}
    for js in judge_samples:
        sub = {sid: js[sid] for sid in common}
        # any judge_name will do; will overwrite per-call
        out[id(js)] = per_judge_stats(sub, "common")
    return common, out


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

def per_category_table(judges: dict[str, dict[str, dict]]) -> dict:
    """For each (judge, category) compute the per-category causal Δ.
    
    Category source: removed_tell_categories (one sample may contribute to
    multiple categories).
    """
    out: dict[str, dict[str, dict]] = {}
    for judge_name, samples in judges.items():
        cats: dict[str, list[dict]] = {}
        for s in samples.values():
            for c in (s.get("removed_tell_categories") or []):
                cats.setdefault(c, []).append(s)
        out[judge_name] = {}
        for c, ss in cats.items():
            base = np.array([s["p_baseline"] for s in ss])
            rem = np.array([s["p_removed"] for s in ss])
            ctrl = np.array([s["p_control"] for s in ss])
            causal = (rem - base) - (ctrl - base)
            out[judge_name][c] = {
                "n": len(ss),
                "base": float(base.mean()),
                "treat": float(rem.mean()),
                "ctrl": float(ctrl.mean()),
                "raw_delta": float((rem - base).mean()),
                "ctrl_delta": float((ctrl - base).mean()),
                "causal_delta": float(causal.mean()),
            }
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def main():
    gpt = load_canonical_per_sample()  # gpt-5-mini via high_prec_subset (needham)
    gemini = load_cross_per_sample(GEMINI)
    qwen = load_cross_per_sample(QWEN)

    print("=" * 102)
    print("P1-3 CROSS-JUDGE REVERSE-INTERVENTION HEADLINE")
    print("Pool: Needham gated-reverse source pool; eligibility = baseline-Peval ≥ 0.5 (per-judge)")
    print("Sign convention: Δ = Peval_after − Peval_before (negative = bank effect)")
    print("=" * 102)
    print()
    print("--- PER-JUDGE HEADLINE -------------------------------------------------------------------------")
    print(f"{'judge':<30} {'n':>4} {'base':>6} {'treat':>6} {'ctrl':>6} "
          f"{'raw Δ':>8} {'ctrl Δ':>8} {'causal Δ':>10} {'CI95':>22} {'d_z':>6} {'p':>10}")
    rows = []
    for name, samples in [("gpt-5-mini (canonical*)", gpt),
                          ("gemini-3-flash-preview", gemini),
                          ("qwen3.5-27b", qwen)]:
        s = per_judge_stats(samples, name)
        rows.append(s)
        if s["n"] == 0:
            print(f"{name:<30} -- no per-sample data --")
            continue
        ci = f"[{s['causal_ci95'][0]:+.3f},{s['causal_ci95'][1]:+.3f}]"
        p_str = "n/a" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.2e}"
        print(f"{name:<30} {s['n']:>4} {s['base']:>6.3f} {s['treat']:>6.3f} {s['ctrl']:>6.3f} "
              f"{s['raw_delta']:>+8.3f} {s['ctrl_delta']:>+8.3f} {s['causal_delta']:>+10.3f} "
              f"{ci:>22} {s['d_z_causal']:>+6.2f} {p_str:>10}")
    print()
    print("* gpt-5-mini row uses the n=154 needham subset of the high-precision rerun")
    print("  (same 5-PQ × 3-epoch protocol as the cross-judge runs, eligibility-filtered)")
    print("  -- comparable to the cross-judge rows. Canonical paper headline (n=93) uses")
    print("  a stricter eligibility (post-baseline + post-hoc QC) and gives Δ=-0.346.")
    print()

    # Direction-consistency
    print("--- DIRECTION-CONSISTENCY (matched-control adjusted) ------------------------------------------")
    print(f"{'judge':<30} {'% dir-pos raw':>15} {'% dir-pos causal':>18} {'raw/ctrl ratio':>15}")
    for s in rows:
        if s["n"] == 0:
            continue
        print(f"{s['judge']:<30} "
              f"{s['pct_dir_pos_raw']*100:>14.1f}% "
              f"{s['pct_dir_pos_causal']*100:>17.1f}% "
              f"{s['ratio_raw_to_ctrl']:>+15.2f}")
    print()

    # Head-to-head on common intersection
    common = set(gpt.keys()) & set(gemini.keys()) & set(qwen.keys())
    print(f"--- HEAD-TO-HEAD ON COMMON INTERSECTION (n = {len(common)}) ---------------------------------")
    print(f"{'judge':<30} {'base':>6} {'treat':>6} {'ctrl':>6} "
          f"{'raw Δ':>8} {'ctrl Δ':>8} {'causal Δ':>10} {'d_z':>6}")
    for name, samples in [("gpt-5-mini (canonical*)", gpt),
                          ("gemini-3-flash-preview", gemini),
                          ("qwen3.5-27b", qwen)]:
        sub = {sid: samples[sid] for sid in common if sid in samples}
        s = per_judge_stats(sub, name)
        if s["n"] == 0:
            continue
        print(f"{name:<30} {s['base']:>6.3f} {s['treat']:>6.3f} {s['ctrl']:>6.3f} "
              f"{s['raw_delta']:>+8.3f} {s['ctrl_delta']:>+8.3f} {s['causal_delta']:>+10.3f} "
              f"{s['d_z_causal']:>+6.2f}")
    print()

    # Per-category
    print("--- PER-CATEGORY CAUSAL Δ ACROSS JUDGES -------------------------------------------------------")
    cat_data = per_category_table({"gpt-5-mini": gpt, "gemini-3-fl-pv": gemini, "qwen3.5-27b": qwen})
    all_cats = sorted({c for j in cat_data.values() for c in j.keys()})
    print(f"{'category':<22} {'gpt-5-mini':>22} {'gemini-3-fl-pv':>22} {'qwen3.5-27b':>22}")
    for cat in all_cats:
        line = [f"{cat:<22}"]
        for j in ["gpt-5-mini", "gemini-3-fl-pv", "qwen3.5-27b"]:
            d = cat_data[j].get(cat)
            if d is None:
                line.append(f"{'-- n=0 --':>22}")
            else:
                line.append(f"  {d['causal_delta']:>+6.3f} (n={d['n']:>3}){'':>4}")
        print(" ".join(line))
    print()

    # Save full structured report
    out_path = REPO / "results" / "cross_judge_full_report.json"
    out_path.write_text(json.dumps({
        "per_judge_headline": [{k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in r.items()} for r in rows],
        "common_intersection_n": len(common),
        "per_category": cat_data,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
