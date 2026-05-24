"""Aggregate per-benchmark probe predictions into a summary table + figure.

Outputs:
  results/summary_table.json  — headline per-benchmark + pooled numbers
  results/summary_table.md    — human-readable markdown table
  results/figure.pdf          — bar chart of P(eval) per condition per benchmark

Usage:
    python -m activation_probing.analyze_probes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config


def bootstrap_ci(values: np.ndarray, n_iter: int = None, ci: float = None) -> tuple[float, float]:
    n_iter = n_iter or config.N_BOOTSTRAP
    ci = ci or config.BOOTSTRAP_CI
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(config.RNG_SEED)
    means = np.empty(n_iter)
    n = len(values)
    for i in range(n_iter):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return (lo, hi)


def load_predictions(benchmark: str) -> dict | None:
    path = config.RESULTS_DIR / f"{benchmark}__predictions.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_summary() -> dict:
    rows = []
    pooled_per_condition = {c: [] for c in config.CONDITIONS}
    for b in config.BENCHMARKS:
        pred = load_predictions(b)
        if not pred:
            continue
        bench_row = {"benchmark": b, "best_layer": pred["best_layer"],
                     "best_layer_cv_acc": pred["best_layer_cv_acc"], "by_condition": {}}
        for cond, payload in pred["per_condition"].items():
            samples = np.array([p["p_eval"] for p in payload["predictions"]])
            lo, hi = bootstrap_ci(samples)
            bench_row["by_condition"][cond] = {
                "n": payload["n"],
                "mean": payload["mean_p_eval"],
                "ci_low": lo,
                "ci_high": hi,
            }
            pooled_per_condition[cond].extend(samples.tolist())
        rows.append(bench_row)

    pooled = {}
    for cond, vals in pooled_per_condition.items():
        if not vals:
            continue
        v = np.array(vals)
        lo, hi = bootstrap_ci(v)
        pooled[cond] = {"n": int(len(v)), "mean": float(v.mean()), "ci_low": lo, "ci_high": hi}

    return {"per_benchmark": rows, "pooled": pooled, "model": config.MODEL_NAME}


def render_markdown(summary: dict) -> str:
    lines = []
    lines.append(f"# Activation Probe Results — {summary['model']}\n")
    lines.append("## Per-benchmark P(eval) by condition\n")
    lines.append("| Benchmark | Layer | CV Acc | " + " | ".join(config.CONDITIONS) + " |")
    lines.append("|" + "---|" * (3 + len(config.CONDITIONS)))
    for row in summary["per_benchmark"]:
        cells = [row["benchmark"], str(row["best_layer"]), f"{row['best_layer_cv_acc']:.2f}"]
        for cond in config.CONDITIONS:
            r = row["by_condition"].get(cond)
            if r:
                cells.append(f"{r['mean']:.2f} (n={r['n']})")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Pooled across benchmarks\n")
    lines.append("| Condition | n | Mean P(eval) | 95% CI |")
    lines.append("|---|---|---|---|")
    for cond in config.CONDITIONS:
        r = summary["pooled"].get(cond)
        if r:
            lines.append(f"| {cond} | {r['n']} | {r['mean']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")

    # Δ vs orig_eval
    lines.append("\n## Pooled Δ vs orig_eval (causal effect of sanitization at activation level)\n")
    base = summary["pooled"].get("orig_eval")
    if base:
        lines.append("| Condition | Δ vs orig_eval |")
        lines.append("|---|---|")
        for cond in config.CONDITIONS:
            if cond == "orig_eval":
                continue
            r = summary["pooled"].get(cond)
            if r:
                d = r["mean"] - base["mean"]
                lines.append(f"| {cond} | {d:+.3f} |")
    return "\n".join(lines) + "\n"


def render_figure(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure")
        return

    benchmarks = [r["benchmark"] for r in summary["per_benchmark"]]
    conditions = config.CONDITIONS
    n_b = len(benchmarks)
    n_c = len(conditions)
    if n_b == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_b * 1.5), 5))
    width = 0.8 / max(1, n_c)
    x = np.arange(n_b)
    for i, cond in enumerate(conditions):
        means = []
        errs = []
        for row in summary["per_benchmark"]:
            r = row["by_condition"].get(cond)
            if r:
                means.append(r["mean"])
                errs.append([(r["mean"] - r["ci_low"]), (r["ci_high"] - r["mean"])])
            else:
                means.append(np.nan)
                errs.append([0, 0])
        means = np.array(means)
        errs = np.array(errs).T
        ax.bar(x + i * width - 0.4 + width / 2, means, width, label=cond, yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15)
    ax.set_ylabel("Probe P(eval)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Activation probe P(eval) per condition\nModel: {summary['model']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  [saved] {out_path}")


def main():
    summary = build_summary()
    (config.RESULTS_DIR / "summary_table.json").write_text(json.dumps(summary, indent=2))
    md = render_markdown(summary)
    (config.RESULTS_DIR / "summary_table.md").write_text(md)
    print(md)
    render_figure(summary, config.RESULTS_DIR / "figure.pdf")


if __name__ == "__main__":
    main()
