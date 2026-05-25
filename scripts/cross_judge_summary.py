"""Build the 3-judge reverse-headline summary table for paper §5 Table 3.

Reads the canonical gpt-5-mini reverse pool (tells/data/reverse/results.json)
and the two cross-judge JSONs written by cross_judge_reverse.py, and emits a
single combined CSV + LaTeX-ready text block that matches the paper's
Table 3 row 1 schema (n, base, treat, ctrl, raw delta, causal delta).

Run as:
    python scripts/cross_judge_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CANONICAL_PATH = REPO / "tells" / "data" / "reverse" / "results.json"
QWEN_PATH = REPO / "results" / "cross_judge_reverse__qwen_qwen3.5-27b.json"
GEMINI_PATH = REPO / "results" / "cross_judge_reverse__google_gemini-3-flash-preview.json"

OUT_CSV = REPO / "results" / "cross_judge_headline.csv"
OUT_TXT = REPO / "results" / "cross_judge_headline.txt"


def _ci(arr):
    if not arr:
        return ""
    return f"[{arr[0]:+.3f}, {arr[1]:+.3f}]"


def load_canonical_all() -> dict:
    rows = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    all_row = next(r for r in rows if r["label"] == "ALL")
    # canonical sign: mean_shift = base - rem (positive = ↓ Peval = bank effect)
    return {
        "judge": "openai/gpt-5-mini",
        "n": all_row["n_paired"],
        "base": all_row["mean_baseline"],
        "treat": all_row["mean_removed"],
        "ctrl": all_row["mean_rev_control"],
        "raw_shift_pos": all_row["mean_shift_removed"],
        "ctrl_shift_pos": all_row["mean_shift_rev_control"],
        "causal_effect_pos": all_row["causal_effect"],
        "causal_ci": "",  # canonical file doesn't ship CI; report from p-values
        "p_removed_lt_revctrl": all_row.get("p_removed_lt_revctrl"),
    }


def load_cross(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    all_row = d["headline_all"]
    return {
        "judge": d["judge"],
        "n": all_row["n_paired"],
        "base": all_row["mean_baseline"],
        "treat": all_row["mean_removed"],
        "ctrl": all_row["mean_rev_control"],
        "raw_shift_pos": all_row["mean_shift_removed"],
        "ctrl_shift_pos": all_row["mean_shift_rev_control"],
        "causal_effect_pos": all_row["causal_effect"],
        "causal_ci": _ci(all_row.get("causal_effect_ci95", [])),
        "p_removed_lt_revctrl": all_row.get("p_removed_lt_revctrl"),
    }


def main():
    rows = []
    rows.append(load_canonical_all())
    gemini = load_cross(GEMINI_PATH)
    qwen = load_cross(QWEN_PATH)
    if gemini:
        rows.append(gemini)
    if qwen:
        rows.append(qwen)

    # Build CSV with paper-sign convention: in §5 the paper writes raw_delta
    # as a NEGATIVE number (Peval after - Peval before = drop). The canonical
    # results.json uses mean_shift = base - rem (POSITIVE = drop). Convert.
    csv_lines = ["judge,n,base,treat,ctrl,raw_delta_paper_sign,ctrl_delta_paper_sign,causal_delta_paper_sign,causal_ci,p_removed_lt_revctrl"]
    txt_lines = []
    txt_lines.append("=" * 100)
    txt_lines.append("Cross-judge reverse-intervention headline (paper Table 3 row 1 extension)")
    txt_lines.append("Pool: Needham gated-reverse source pool, eligibility-filtered (baseline >= 0.5)")
    txt_lines.append("Sign convention: paper §1 (Delta = Peval_after - Peval_before, negative = bank effect)")
    txt_lines.append("=" * 100)
    txt_lines.append(f"{'judge':<35} {'n':>4} {'base':>6} {'treat':>6} {'ctrl':>6} "
                     f"{'raw Δ':>8} {'ctrl Δ':>8} {'causal Δ':>10} {'causal CI95':>22} {'p':>10}")
    txt_lines.append("-" * 100)
    for r in rows:
        # paper sign: raw_delta = treat - base = -mean_shift_pos
        raw_paper = -r["raw_shift_pos"]
        ctrl_paper = -r["ctrl_shift_pos"]
        causal_paper = -r["causal_effect_pos"]
        # CI from script is on shift_pos - ctrl_pos = bank-specific drop; flip sign
        ci_paper = ""
        if r["causal_ci"]:
            # Parse "[lo, hi]" and flip+swap
            try:
                lo, hi = (float(x.strip()) for x in r["causal_ci"].strip("[]").split(","))
                ci_paper = f"[{-hi:+.3f}, {-lo:+.3f}]"
            except Exception:
                ci_paper = r["causal_ci"]
        p = r.get("p_removed_lt_revctrl")
        p_str = "n/a" if p is None else f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
        txt_lines.append(
            f"{r['judge']:<35} {r['n']:>4} {r['base']:>6.3f} {r['treat']:>6.3f} {r['ctrl']:>6.3f} "
            f"{raw_paper:>+8.3f} {ctrl_paper:>+8.3f} {causal_paper:>+10.3f} {ci_paper:>22} {p_str:>10}"
        )
        csv_lines.append(
            f"{r['judge']},{r['n']},{r['base']:.4f},{r['treat']:.4f},{r['ctrl']:.4f},"
            f"{raw_paper:+.4f},{ctrl_paper:+.4f},{causal_paper:+.4f},\"{ci_paper}\",{p_str}"
        )

    txt_lines.append("=" * 100)
    txt_lines.append("")
    txt_lines.append("Notes for paper write-up:")
    txt_lines.append("  - Direction (negative causal Δ) is the model-agnostic claim.")
    txt_lines.append("  - Absolute magnitude scales with baseline calibration on the")
    txt_lines.append("    E-case pool (gpt-5-mini base=0.90; alternatives often saturate near 1.0).")
    txt_lines.append("  - Reduced cross-judge magnitude is the expected outcome under the")
    txt_lines.append("    paper's 'Absolute Δ magnitudes are judge-specific' paragraph (§4/§5).")

    print("\n".join(txt_lines))
    OUT_TXT.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    OUT_CSV.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_TXT}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
