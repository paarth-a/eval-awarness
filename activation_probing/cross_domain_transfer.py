"""(2) Cross-domain probe transfer.

Train probe on CAPABILITY-only data (needham) and test on SAFETY (hawthorne, mask)
and vice versa. If accuracy collapses, the probe is domain-specific — safety detector
or capability detector, not eval-awareness.

Also train two pooled probes:
  - capability-pool: same as needham-only since needham is the only capability benchmark with D-cases
  - safety-pool: union of hawthorne + mask
Then cross-evaluate.
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


CAP = ["needham"]              # capability domain (the only one with D-cases)
SAFETY = ["hawthorne", "mask"]  # safety domain


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    return {int(li): z[f"layer_{li}"] for li in layer_indices}


def train_pooled(benchmarks, layer):
    """Train pooled probe over given benchmarks at given layer. Returns (scaler, clf, cv)."""
    pos_list, neg_list = [], []
    for b in benchmarks:
        oe = load_acts(cache_path(b, "orig_eval"))
        od = load_acts(cache_path(b, "orig_deploy"))
        if layer in oe and layer in od:
            pos_list.append(oe[layer])
            neg_list.append(od[layer])
    if not pos_list:
        return None
    pos = np.concatenate(pos_list, axis=0)
    neg = np.concatenate(neg_list, axis=0)
    X = np.concatenate([pos, neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    cv = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy").mean()
    clf_final = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    clf_final.fit(Xs, y)
    return scaler, clf_final, float(cv), len(pos), len(neg)


def eval_on_benchmarks(probe, benchmarks, layer):
    """Apply probe to each benchmark's E vs D, return per-benchmark AUC."""
    scaler, clf, cv, _, _ = probe
    results = {}
    for b in benchmarks:
        oe = load_acts(cache_path(b, "orig_eval"))
        od = load_acts(cache_path(b, "orig_deploy"))
        if layer not in oe or layer not in od:
            continue
        X = np.concatenate([oe[layer], od[layer]], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(oe[layer])), np.zeros(len(od[layer]))])
        proba = clf.predict_proba(scaler.transform(X))[:, 1]
        try:
            auc = float(roc_auc_score(y, proba))
        except Exception:
            auc = float("nan")
        results[b] = {"auc": auc, "n_eval": int(len(oe[layer])), "n_deploy": int(len(od[layer]))}
    return results


def main():
    print("Model:", config.MODEL_NAME)
    # Find shared layer across all benchmarks
    needed = CAP + SAFETY
    shared = None
    for b in needed:
        oe = load_acts(cache_path(b, "orig_eval"))
        od = load_acts(cache_path(b, "orig_deploy"))
        layers = set(oe.keys()) & set(od.keys())
        shared = layers if shared is None else shared & layers
    if not shared:
        print("No shared layers, abort")
        return

    # Pick representative layer — use 16 (middle layer commonly available)
    layer = 16 if 16 in shared else min(shared, key=lambda l: abs(l-16))
    print(f"Using shared layer: {layer}")
    print()

    # Train cap-only probe
    print("Training CAPABILITY-only probe (trained on needham E vs D)...")
    cap_probe = train_pooled(CAP, layer)
    print(f"  CV acc: {cap_probe[2]:.3f}  (n_pos={cap_probe[3]}, n_neg={cap_probe[4]})")

    # Train safety-only probe
    print("Training SAFETY-only probe (trained on hawthorne+mask E vs D)...")
    safe_probe = train_pooled(SAFETY, layer)
    print(f"  CV acc: {safe_probe[2]:.3f}  (n_pos={safe_probe[3]}, n_neg={safe_probe[4]})")

    print()
    print("=" * 80)
    print("(2) Cross-DOMAIN probe transfer (out-of-domain AUC)")
    print("-" * 80)
    print(f"{'probe trained on':<30} {'target':<14} {'AUC':>8} {'n_e':>5} {'n_d':>5}")

    # Cap probe → safety domain
    cap_on_safe = eval_on_benchmarks(cap_probe, SAFETY, layer)
    for b, r in cap_on_safe.items():
        print(f"{'capability (needham)':<30} {b:<14} {r['auc']:>8.3f} {r['n_eval']:>5} {r['n_deploy']:>5}")

    # Cap probe → cap (self, sanity)
    cap_on_cap = eval_on_benchmarks(cap_probe, CAP, layer)
    for b, r in cap_on_cap.items():
        print(f"{'capability (needham)':<30} {b:<14} {r['auc']:>8.3f} {r['n_eval']:>5} {r['n_deploy']:>5} [SELF]")

    print()
    # Safety probe → capability domain
    safe_on_cap = eval_on_benchmarks(safe_probe, CAP, layer)
    for b, r in safe_on_cap.items():
        print(f"{'safety (hawthorne+mask)':<30} {b:<14} {r['auc']:>8.3f} {r['n_eval']:>5} {r['n_deploy']:>5}")

    # Safety probe → safety (self, sanity)
    safe_on_safe = eval_on_benchmarks(safe_probe, SAFETY, layer)
    for b, r in safe_on_safe.items():
        print(f"{'safety (hawthorne+mask)':<30} {b:<14} {r['auc']:>8.3f} {r['n_eval']:>5} {r['n_deploy']:>5} [SELF]")

    # Summary
    cap_to_safety_aucs = [r["auc"] for r in cap_on_safe.values()]
    safe_to_cap_aucs = [r["auc"] for r in safe_on_cap.values()]
    print()
    print(f"capability→safety mean AUC: {np.mean(cap_to_safety_aucs):.3f}")
    print(f"safety→capability mean AUC: {np.mean(safe_to_cap_aucs):.3f}")
    print()
    cap_self_aucs = [r["auc"] for r in cap_on_cap.values()]
    safe_self_aucs = [r["auc"] for r in safe_on_safe.values()]
    print(f"capability self AUC: {np.mean(cap_self_aucs):.3f}")
    print(f"safety self AUC:     {np.mean(safe_self_aucs):.3f}")

    print()
    print("VERDICT:")
    cap_drop = np.mean(cap_self_aucs) - np.mean(cap_to_safety_aucs)
    safe_drop = np.mean(safe_self_aucs) - np.mean(safe_to_cap_aucs)
    if max(cap_drop, safe_drop) > 0.25:
        print(f"  Probe is DOMAIN-SPECIFIC.")
        print(f"  capability self {np.mean(cap_self_aucs):.3f} - cap→safe {np.mean(cap_to_safety_aucs):.3f} = drop {cap_drop:.3f}")
        print(f"  safety self     {np.mean(safe_self_aucs):.3f} - safe→cap {np.mean(safe_to_cap_aucs):.3f} = drop {safe_drop:.3f}")
        if cap_to_safety_aucs and np.mean(cap_to_safety_aucs) < 0.65:
            print("  -> capability-trained probe FAILS on safety. Probe is partly content-specific.")
        if safe_to_cap_aucs and np.mean(safe_to_cap_aucs) < 0.65:
            print("  -> safety-trained probe FAILS on capability. Probe is partly content-specific.")
    else:
        print(f"  Probe is CROSS-DOMAIN. Drops are small ({cap_drop:.3f}, {safe_drop:.3f}).")
        print(f"  This rules out 'safety detector' or 'capability detector' as primary signal.")

    out = config.RESULTS_DIR / "cross_domain_transfer.json"
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "layer": int(layer),
        "cap_probe_cv": cap_probe[2],
        "safe_probe_cv": safe_probe[2],
        "cap_to_safety": cap_on_safe,
        "cap_self": cap_on_cap,
        "safe_to_cap": safe_on_cap,
        "safe_self": safe_on_safe,
        "mean_cap_to_safety_auc": float(np.mean(cap_to_safety_aucs)) if cap_to_safety_aucs else None,
        "mean_safe_to_cap_auc": float(np.mean(safe_to_cap_aucs)) if safe_to_cap_aucs else None,
    }, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
