"""M2: pooled-E-vs-pooled-D probe.

Train one probe on the union of all benchmarks' orig_eval (positives) and
orig_deploy (negatives). Apply to each benchmark's v3_removed, compare Δ to
per-benchmark probes.

If pooled-probe v3 Δ ≈ per-benchmark v3 Δ → both probes capture similar signal
(eval-vs-deploy is a general axis).

If pooled-probe v3 Δ ≪ per-benchmark Δ → per-benchmark probes were over-
specialized (overfitting to benchmark identity).

Usage:
    python -m activation_probing.m2_pooled_probe
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path

import os
PROBES_DIR_OVERRIDE = os.environ.get("PROBES_DIR")
if PROBES_DIR_OVERRIDE:
    config.PROBES_DIR = Path(PROBES_DIR_OVERRIDE)
RESULTS_DIR_OVERRIDE = os.environ.get("RESULTS_DIR")
if RESULTS_DIR_OVERRIDE:
    config.RESULTS_DIR = Path(RESULTS_DIR_OVERRIDE)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BENCHMARKS_WITH_BOTH = ["needham", "hawthorne", "mask"]


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    return {int(li): z[f"layer_{li}"] for li in layer_indices}


def main():
    print("Model:", config.MODEL_NAME)

    # Find shared layers across all benchmarks' orig_eval and orig_deploy
    all_shared = None
    per_b_acts = {}
    for b in BENCHMARKS_WITH_BOTH:
        oe = cache_path(b, "orig_eval")
        od = cache_path(b, "orig_deploy")
        if not (oe.exists() and od.exists()):
            print(f"  [skip] {b}: missing orig_eval or orig_deploy")
            continue
        oe_a = load_acts(oe)
        od_a = load_acts(od)
        layers = set(oe_a.keys()) & set(od_a.keys())
        per_b_acts[b] = {"oe": oe_a, "od": od_a}
        all_shared = layers if all_shared is None else all_shared & layers
    if not per_b_acts or not all_shared:
        print("No shared layers across benchmarks. Cannot train pooled probe.")
        return

    shared_layers = sorted(all_shared)
    print(f"Benchmarks with both E and D: {list(per_b_acts.keys())}")
    print(f"Shared layers: {shared_layers}")

    # Per-layer CV on POOLED data
    print()
    print("=" * 70)
    print("M2 — Pooled probe: per-layer CV accuracy")
    print("-" * 70)
    print(f"{'layer':>6} {'cv_acc':>10} {'n_pos':>8} {'n_neg':>8}")
    per_layer_stats = {}
    for li in shared_layers:
        pos = np.concatenate([per_b_acts[b]["oe"][li] for b in per_b_acts], axis=0)
        neg = np.concatenate([per_b_acts[b]["od"][li] for b in per_b_acts], axis=0)
        X = np.concatenate([pos, neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        if X.shape[0] < 2 * config.PROBE_CV_FOLDS:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=config.PROBE_CV_FOLDS, shuffle=True,
                              random_state=config.RNG_SEED)
        clf = LogisticRegression(C=config.PROBE_C, max_iter=2000, random_state=config.RNG_SEED,
                                 class_weight="balanced")
        scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy")
        per_layer_stats[li] = {
            "cv_acc": float(scores.mean()),
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
        }
        print(f"{li:>6} {scores.mean():>10.3f} {len(pos):>8d} {len(neg):>8d}")

    # Pick best layer
    best_layer = max(per_layer_stats.keys(), key=lambda li: per_layer_stats[li]["cv_acc"])
    print()
    print(f"Best pooled layer: {best_layer} (cv_acc={per_layer_stats[best_layer]['cv_acc']:.3f})")

    # Train final pooled probe at best layer
    pos = np.concatenate([per_b_acts[b]["oe"][best_layer] for b in per_b_acts], axis=0)
    neg = np.concatenate([per_b_acts[b]["od"][best_layer] for b in per_b_acts], axis=0)
    X = np.concatenate([pos, neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf_pool = LogisticRegression(C=config.PROBE_C, max_iter=2000,
                                  random_state=config.RNG_SEED, class_weight="balanced")
    clf_pool.fit(Xs, y)

    # Apply pooled probe to v3_removed for each benchmark; compare to per-benchmark probe
    print()
    print("=" * 70)
    print("M2 — Pooled vs per-benchmark probe Δ comparison")
    print("(orig_eval P → v3_removed P)")
    print("-" * 70)
    print(f"{'benchmark':<12} {'P(eval) orig':>14} {'P(eval) v3':>14} {'pooled Δ':>14} {'per-bench Δ':>14}")

    results = {}
    for b in BENCHMARKS_WITH_BOTH:
        oe_path = cache_path(b, "orig_eval")
        v3_path = cache_path(b, "v3_removed")
        if not (oe_path.exists() and v3_path.exists()):
            continue
        oe_a = load_acts(oe_path)
        v3_a = load_acts(v3_path)
        if best_layer not in oe_a or best_layer not in v3_a:
            continue

        # Pooled probe predictions
        oe_X = scaler.transform(oe_a[best_layer])
        v3_X = scaler.transform(v3_a[best_layer])
        oe_p_pool = float(clf_pool.predict_proba(oe_X)[:, 1].mean())
        v3_p_pool = float(clf_pool.predict_proba(v3_X)[:, 1].mean())
        pool_delta = v3_p_pool - oe_p_pool

        # Per-benchmark probe Δ (load existing pickle)
        per_b_probe_path = config.PROBES_DIR / f"{b}__probes.pkl"
        per_b_delta = None
        if per_b_probe_path.exists():
            with open(per_b_probe_path, "rb") as f:
                pb = pickle.load(f)
            pb_layer = int(pb["best_layer"])
            if pb_layer in oe_a and pb_layer in v3_a:
                pb_scaler = pb["scaler"]
                pb_clf = pb["classifier"]
                pb_oe = float(pb_clf.predict_proba(pb_scaler.transform(oe_a[pb_layer]))[:, 1].mean())
                pb_v3 = float(pb_clf.predict_proba(pb_scaler.transform(v3_a[pb_layer]))[:, 1].mean())
                per_b_delta = pb_v3 - pb_oe

        results[b] = {
            "pool_oe_p": oe_p_pool,
            "pool_v3_p": v3_p_pool,
            "pool_delta": pool_delta,
            "per_bench_delta": per_b_delta,
        }
        pb_str = f"{per_b_delta:>+14.3f}" if per_b_delta is not None else " " * 14
        print(f"{b:<12} {oe_p_pool:>14.3f} {v3_p_pool:>14.3f} {pool_delta:>+14.3f} {pb_str}")

    # Pooled
    pool_deltas = [r["pool_delta"] for r in results.values()]
    per_b_deltas = [r["per_bench_delta"] for r in results.values() if r["per_bench_delta"] is not None]
    print()
    print(f"Mean pooled Δ:       {np.mean(pool_deltas):>+.3f}  (n={len(pool_deltas)})")
    if per_b_deltas:
        print(f"Mean per-bench Δ:    {np.mean(per_b_deltas):>+.3f}  (n={len(per_b_deltas)})")
        print(f"Ratio pooled/per-b:  {np.mean(pool_deltas)/np.mean(per_b_deltas):.2f}x")
    print()
    if pool_deltas:
        if abs(np.mean(pool_deltas)) >= 0.7 * abs(np.mean(per_b_deltas) if per_b_deltas else 1.0):
            print("VERDICT: Pooled probe shows comparable Δ to per-benchmark probes.")
            print("  → The bank-removal signal generalizes; per-bench probes weren't just overfitting.")
        else:
            print("VERDICT: Pooled probe Δ is substantially smaller than per-benchmark.")
            print("  → Per-bench probes were partly overfitting to benchmark identity.")

    # Save
    out = config.RESULTS_DIR / "m2_pooled_probe.json"
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "benchmarks_used": list(per_b_acts.keys()),
        "best_layer": int(best_layer),
        "best_layer_cv_acc": per_layer_stats[best_layer]["cv_acc"],
        "per_layer_cv": {int(li): per_layer_stats[li]["cv_acc"] for li in per_layer_stats},
        "per_benchmark_v3_delta": {k: v for k, v in results.items()},
        "mean_pooled_delta": float(np.mean(pool_deltas)) if pool_deltas else None,
        "mean_per_bench_delta": float(np.mean(per_b_deltas)) if per_b_deltas else None,
    }, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
