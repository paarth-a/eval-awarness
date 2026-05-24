"""Compute per-layer CV accuracy for an existing model's cached activations.

For each benchmark with both orig_eval and orig_deploy cached, computes
logistic regression CV accuracy at every captured layer. Outputs JSON and
a simple markdown table for use in the paper.

Usage:
    ACTPROBE_MODEL=Qwen/Qwen3.5-122B-A10B python -m activation_probing.per_layer_curve
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    return {int(li): z[f"layer_{li}"] for li in layer_indices}


def per_layer_cv(X_eval, X_deploy, folds=5):
    X = np.concatenate([X_eval, X_deploy], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(X_eval), dtype=np.int64),
                        np.zeros(len(X_deploy), dtype=np.int64)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    scores = cross_val_score(clf, Xs, y, cv=folds, scoring="accuracy")
    return float(np.mean(scores)), float(np.std(scores))


def main():
    model = config.MODEL_NAME
    print("Model:", model)
    benchmarks = ["needham", "hawthorne", "mask"]

    summary = {}
    for b in benchmarks:
        oe_path = cache_path(b, "orig_eval")
        od_path = cache_path(b, "orig_deploy")
        if not (oe_path.exists() and od_path.exists()):
            print(f"[skip] {b}: missing orig_eval or orig_deploy cache")
            continue
        oe_acts = load_acts(oe_path)
        od_acts = load_acts(od_path)
        common_layers = sorted(set(oe_acts.keys()) & set(od_acts.keys()))
        print(f"\n=== {b} (n_eval={len(next(iter(oe_acts.values())))}, n_deploy={len(next(iter(od_acts.values())))}, layers={common_layers}) ===")
        rows = []
        for li in common_layers:
            mean, std = per_layer_cv(oe_acts[li], od_acts[li])
            rows.append((li, mean, std))
            print("  layer %3d: cv_acc = %.3f +/- %.3f" % (li, mean, std))
        rows.sort(key=lambda r: -r[1])
        best = rows[0]
        print("  best: layer %d (%.3f)" % (best[0], best[1]))
        summary[b] = {
            "layers": [{"layer": li, "cv_acc_mean": m, "cv_acc_std": s} for li, m, s in sorted(rows)],
            "best_layer": best[0],
            "best_cv_acc": best[1],
        }

    # Markdown table
    print()
    print("=" * 80)
    print("PER-LAYER CV ACC SUMMARY for", model)
    print("-" * 80)
    all_layers = sorted({li for b in summary for li in [r["layer"] for r in summary[b]["layers"]]})
    hdr = "layer  | " + " | ".join("%9s" % b for b in summary.keys())
    print(hdr)
    print("-" * len(hdr))
    for li in all_layers:
        cells = []
        for b in summary.keys():
            vals = {r["layer"]: r["cv_acc_mean"] for r in summary[b]["layers"]}
            if li in vals:
                cells.append("%9.3f" % vals[li])
            else:
                cells.append("        -")
        print("%-6d | %s" % (li, " | ".join(cells)))

    out_json = config.RESULTS_DIR / f"per_layer_cv_{model.replace('/', '__')}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
