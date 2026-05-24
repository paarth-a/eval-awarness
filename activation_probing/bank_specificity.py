"""Apply pooled probe to orig_eval, v2/v3_removed, rev_control for hawthorne and SAD.
Compute bank-specificity ratio = (orig - v3) / (orig - rev_control).
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    return {int(li): z[f"layer_{li}"] for li in list(z["layer_indices"])}


def main():
    print("Model:", config.MODEL_NAME)
    train_b = ["needham", "hawthorne", "mask"]
    per_b_acts = {}
    for b in train_b:
        oe = cache_path(b, "orig_eval")
        od = cache_path(b, "orig_deploy")
        if oe.exists() and od.exists():
            per_b_acts[b] = {"oe": load_acts(oe), "od": load_acts(od)}
    layer_sets = [set(per_b_acts[b]["oe"].keys()) & set(per_b_acts[b]["od"].keys()) for b in per_b_acts]
    shared = sorted(set.intersection(*layer_sets))

    # Pick best layer for pooled probe (CV across pooled data)
    best_layer = None
    best_cv = -1
    for li in shared:
        pos = np.concatenate([per_b_acts[b]["oe"][li] for b in per_b_acts], axis=0)
        neg = np.concatenate([per_b_acts[b]["od"][li] for b in per_b_acts], axis=0)
        X = np.concatenate([pos, neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        if X.shape[0] < 25:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
        cv = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy").mean()
        if cv > best_cv:
            best_cv = cv
            best_layer = li
    print(f"Pooled probe best layer: {best_layer} (CV {best_cv:.3f})")

    # Train final pooled probe at best layer
    pos = np.concatenate([per_b_acts[b]["oe"][best_layer] for b in per_b_acts], axis=0)
    neg = np.concatenate([per_b_acts[b]["od"][best_layer] for b in per_b_acts], axis=0)
    X = np.concatenate([pos, neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(Xs, y)

    # Apply to each target benchmark's conditions
    print()
    print("=" * 90)
    print("Bank-specificity (pooled probe). orig_eval → v3_removed (or v2 for SAD) → rev_control")
    print("-" * 90)
    print(f"{'benchmark':<12} {'orig P(eval)':>14} {'v3/v2 P(eval)':>15} {'rev_ctrl P(eval)':>17} {'v Δ':>9} {'ctrl Δ':>9} {'ratio':>10}")

    results = {}
    for b, cond_ablation in [("hawthorne", "v3_removed"), ("sad", "v2_removed")]:
        try:
            oe_p = cache_path(b, "orig_eval")
            ab_p = cache_path(b, cond_ablation)
            rc_p = cache_path(b, "rev_control")
            if not all(p.exists() for p in (oe_p, ab_p, rc_p)):
                print(f"{b:<12} missing one of: {oe_p.exists()=} {ab_p.exists()=} {rc_p.exists()=}")
                continue
            oe = load_acts(oe_p)
            ab = load_acts(ab_p)
            rc = load_acts(rc_p)
            if best_layer not in oe or best_layer not in ab or best_layer not in rc:
                print(f"{b:<12} best layer {best_layer} not cached in all 3 cells")
                continue
            oe_pred = clf.predict_proba(scaler.transform(oe[best_layer]))[:, 1].mean()
            ab_pred = clf.predict_proba(scaler.transform(ab[best_layer]))[:, 1].mean()
            rc_pred = clf.predict_proba(scaler.transform(rc[best_layer]))[:, 1].mean()
            v_delta = float(ab_pred - oe_pred)
            c_delta = float(rc_pred - oe_pred)
            ratio = (v_delta / c_delta) if c_delta != 0 else float("nan")
            results[b] = {
                "ablation_cond": cond_ablation,
                "orig_p_eval": float(oe_pred),
                "ablated_p_eval": float(ab_pred),
                "rev_control_p_eval": float(rc_pred),
                "ablation_delta": v_delta,
                "rev_control_delta": c_delta,
                "bank_specificity_ratio": ratio,
            }
            print(f"{b:<12} {oe_pred:>14.3f} {ab_pred:>15.3f} {rc_pred:>17.3f} {v_delta:>+9.3f} {c_delta:>+9.3f} {ratio:>10.2f}x")
        except Exception as e:
            print(f"{b}: error {e}")

    out = config.RESULTS_DIR / "bank_specificity_pooled.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "pooled_best_layer": int(best_layer),
        "pooled_cv": float(best_cv),
        "results": results,
    }, indent=2))
    print(f"\n[saved] {out}")

    print()
    print("INTERPRETATION:")
    print("  ratio > 1: bank-specific (bank removal moves probe more than matched random rewrite)")
    print("  ratio ≈ 1: bank-effect equal to random-rewrite-of-same-scope (probe is scope-sensitive, not bank-specific)")
    print("  ratio < 0: opposite-direction effect from controls")


if __name__ == "__main__":
    main()
