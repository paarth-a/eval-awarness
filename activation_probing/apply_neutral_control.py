"""Apply trained probe to rev_control activations and compare to v3 results.

Usage:
    python -m activation_probing.apply_neutral_control
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    return ({li: z[f"layer_{li}"] for li in layer_indices}, list(z["sample_ids"]))


def apply_for_benchmark(benchmark):
    probe_path = config.PROBES_DIR / f"{benchmark}__probes.pkl"
    if not probe_path.exists():
        return None
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    best_layer = int(probe["best_layer"])
    scaler = probe["scaler"]
    clf = probe["classifier"]

    out = {"benchmark": benchmark, "model": config.MODEL_NAME, "best_layer": best_layer,
           "best_layer_cv_acc": float(probe["best_layer_cv_acc"]), "per_condition": {}}
    for cond in ["orig_eval", "orig_deploy", "v1_removed", "v2_removed", "v3_removed", "rev_control"]:
        p = cache_path(benchmark, cond)
        if not p.exists():
            continue
        acts, sample_ids = load_acts(p)
        if best_layer not in acts:
            continue
        X = acts[best_layer]
        Xs = scaler.transform(X)
        proba = clf.predict_proba(Xs)[:, 1]
        out["per_condition"][cond] = {
            "n": int(len(proba)),
            "mean_p_eval": float(proba.mean()),
            "median_p_eval": float(np.median(proba)),
            "std_p_eval": float(proba.std()),
        }
    return out


def main():
    print(f"Model: {config.MODEL_NAME}")
    print()
    hdr = "%-12s %10s %10s %10s %8s %8s %10s" % ("benchmark", "orig_eval", "rev_ctrl", "v3_remove", "v3 D", "ctrl D", "spec_ratio")
    print(hdr)
    print("-" * 80)
    pooled_orig, pooled_v3, pooled_ctrl = [], [], []
    for b in ["needham", "hawthorne", "mask", "agentharm", "bbh", "malt"]:
        r = apply_for_benchmark(b)
        if not r:
            continue
        pc = r["per_condition"]
        oe = pc.get("orig_eval", {}).get("mean_p_eval")
        v3 = pc.get("v3_removed", {}).get("mean_p_eval")
        ctrl = pc.get("rev_control", {}).get("mean_p_eval")
        v3d = (v3 - oe) if (v3 is not None and oe is not None) else None
        ctrld = (ctrl - oe) if (ctrl is not None and oe is not None) else None
        ratio = (v3d / ctrld) if (v3d is not None and ctrld and ctrld != 0) else None
        line = "%-12s %10.3f %10.3f %10.3f %+8.3f %+8.3f %10.2f" % (
            b, oe or 0, ctrl or 0, v3 or 0, v3d or 0, ctrld or 0, ratio or 0)
        print(line)
        out_path = config.RESULTS_DIR / f"{b}__predictions_with_control.json"
        out_path.write_text(json.dumps(r, indent=2))
        if oe is not None and v3 is not None and ctrl is not None:
            pooled_orig.append((oe, pc["orig_eval"]["n"]))
            pooled_v3.append((v3, pc["v3_removed"]["n"]))
            pooled_ctrl.append((ctrl, pc["rev_control"]["n"]))

    if pooled_orig and pooled_v3 and pooled_ctrl:
        def wm(items):
            total_n = sum(n for _, n in items)
            return sum(v * n for v, n in items) / total_n if total_n else 0
        oe = wm(pooled_orig)
        v3 = wm(pooled_v3)
        ctrl = wm(pooled_ctrl)
        print("-" * 80)
        print("%-12s %10.3f %10.3f %10.3f %+8.3f %+8.3f" % (
            "POOLED", oe, ctrl, v3, v3 - oe, ctrl - oe))
        print()
        if ctrl - oe != 0:
            print("v3 Δ vs ctrl Δ ratio: %.2fx (>>1 means v3 effect is bank-specific not random)" % ((v3 - oe) / (ctrl - oe)))


if __name__ == "__main__":
    main()
