"""EXP B: apply pooled + per-benchmark probes to SAD activations.

SAD has orig_eval and v2_removed cached (no orig_deploy since SAD is eval-only,
no v3 yet). We apply: (1) each per-benchmark probe (needham, hawthorne, mask)
to SAD orig_eval and SAD v2_removed; (2) a pooled probe trained on the union.

Compute Δ = P(eval) on v2_removed − P(eval) on orig_eval.

Pooled-probe Δ on SAD is the M2-style cleaner test: does the bank-removal
effect generalize to a benchmark NEVER seen in training?
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    sad_oe_path = cache_path("sad", "orig_eval")
    sad_v2_path = cache_path("sad", "v2_removed")
    if not (sad_oe_path.exists() and sad_v2_path.exists()):
        print(f"SAD cache missing: {sad_oe_path} / {sad_v2_path}")
        return
    sad_oe = load_acts(sad_oe_path)
    sad_v2 = load_acts(sad_v2_path)
    print(f"SAD orig_eval cells: {len(next(iter(sad_oe.values())))}")
    print(f"SAD v2_removed cells: {len(next(iter(sad_v2.values())))}")

    rows = []

    # Per-benchmark probes
    for b in BENCHMARKS_WITH_BOTH:
        probe_path = config.PROBES_DIR / f"{b}__probes.pkl"
        if not probe_path.exists():
            continue
        with open(probe_path, "rb") as f:
            probe = pickle.load(f)
        layer = int(probe["best_layer"])
        if layer not in sad_oe or layer not in sad_v2:
            continue
        scaler = probe["scaler"]
        clf = probe["classifier"]
        oe_p = float(clf.predict_proba(scaler.transform(sad_oe[layer]))[:, 1].mean())
        v2_p = float(clf.predict_proba(scaler.transform(sad_v2[layer]))[:, 1].mean())
        rows.append({
            "probe": f"{b} (per-benchmark)",
            "layer": layer,
            "sad_oe_p_eval": oe_p,
            "sad_v2_p_eval": v2_p,
            "delta": v2_p - oe_p,
        })
        print(f"  per-bench {b:<10} probe(layer {layer}): SAD oe={oe_p:.3f} v2={v2_p:.3f} Δ={v2_p-oe_p:+.3f}")

    # Pooled probe: train on union, apply to SAD
    print()
    print("Training pooled probe on union(needham, hawthorne, mask) E vs D ...")
    per_b_acts = {}
    for b in BENCHMARKS_WITH_BOTH:
        oe = cache_path(b, "orig_eval")
        od = cache_path(b, "orig_deploy")
        if not (oe.exists() and od.exists()):
            continue
        per_b_acts[b] = {"oe": load_acts(oe), "od": load_acts(od)}

    if not per_b_acts:
        print("No training data for pooled probe; skipping.")
    else:
        layer_sets = [set(per_b_acts[b]["oe"].keys()) & set(per_b_acts[b]["od"].keys()) for b in per_b_acts]
        shared_layers = sorted(set.intersection(*layer_sets) & set(sad_oe.keys()) & set(sad_v2.keys()))

        from sklearn.model_selection import cross_val_score, StratifiedKFold
        best_layer = None
        best_cv = -1
        per_layer_cv = {}
        for li in shared_layers:
            pos = np.concatenate([per_b_acts[b]["oe"][li] for b in per_b_acts], axis=0)
            neg = np.concatenate([per_b_acts[b]["od"][li] for b in per_b_acts], axis=0)
            X = np.concatenate([pos, neg], axis=0).astype(np.float32)
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            if X.shape[0] < 10:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
            cv = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy").mean()
            per_layer_cv[li] = float(cv)
            if cv > best_cv:
                best_cv = cv
                best_layer = li

        print(f"Best pooled layer: {best_layer} (cv_acc={best_cv:.3f})")
        # Train final pooled at best layer
        pos = np.concatenate([per_b_acts[b]["oe"][best_layer] for b in per_b_acts], axis=0)
        neg = np.concatenate([per_b_acts[b]["od"][best_layer] for b in per_b_acts], axis=0)
        X = np.concatenate([pos, neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf_pool = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
        clf_pool.fit(Xs, y)

        oe_p = float(clf_pool.predict_proba(scaler.transform(sad_oe[best_layer]))[:, 1].mean())
        v2_p = float(clf_pool.predict_proba(scaler.transform(sad_v2[best_layer]))[:, 1].mean())
        rows.append({
            "probe": "POOLED (union: needham+hawthorne+mask)",
            "layer": int(best_layer),
            "sad_oe_p_eval": oe_p,
            "sad_v2_p_eval": v2_p,
            "delta": v2_p - oe_p,
        })
        print(f"\n  POOLED probe(layer {best_layer}): SAD oe={oe_p:.3f} v2={v2_p:.3f} Δ={v2_p-oe_p:+.3f}")

    # Save
    out = config.RESULTS_DIR / "expB_sad_pooled.json"
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "results": rows,
    }, indent=2))
    print()
    print(f"[saved] {out}")

    print()
    print("=" * 70)
    print("EXP B — SAD bank-ablation effect (v2 vs orig)")
    print("-" * 70)
    print(f"{'probe':<45} {'SAD oe':>8} {'SAD v2':>8} {'Δ':>10}")
    for r in rows:
        print(f"{r['probe']:<45} {r['sad_oe_p_eval']:>8.3f} {r['sad_v2_p_eval']:>8.3f} {r['delta']:>+10.3f}")


if __name__ == "__main__":
    main()
