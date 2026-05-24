"""Cross-apply each trained probe to ALL benchmarks (including those without their own probe).

For each (probe_benchmark, target_benchmark, condition) combo, compute mean P(eval).
Used to test bank-specificity on benchmarks lacking their own probe (BBH, MALT, AgentHarm).

Usage:
    python -m activation_probing.cross_apply
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
    return {li: z[f"layer_{li}"] for li in layer_indices}


def apply_probe(probe, acts_map):
    best_layer = int(probe["best_layer"])
    if best_layer not in acts_map:
        return None
    X = acts_map[best_layer]
    Xs = probe["scaler"].transform(X)
    proba = probe["classifier"].predict_proba(Xs)[:, 1]
    return float(proba.mean()), int(len(proba))


def main():
    print("Model:", config.MODEL_NAME)
    benchmarks = ["needham", "hawthorne", "mask", "agentharm", "bbh", "malt"]
    conditions = ["orig_eval", "orig_deploy", "v1_removed", "v2_removed", "v3_removed", "rev_control"]
    probes = {}
    for b in ["needham", "hawthorne", "mask"]:
        p = config.PROBES_DIR / f"{b}__probes.pkl"
        if p.exists():
            with open(p, "rb") as f:
                probes[b] = pickle.load(f)
    print("Loaded probes:", list(probes.keys()))
    print()

    for probe_b in probes:
        probe = probes[probe_b]
        layer = probe["best_layer"]
        cv = probe["best_layer_cv_acc"]
        print("=" * 95)
        print("PROBE: %s (layer %d, CV acc %.3f)" % (probe_b, layer, cv))
        print("-" * 95)
        hdr = "%-12s %10s %10s %10s %10s %10s %10s" % ("target", "orig_eval", "orig_dpl", "v1_rem", "v2_rem", "v3_rem", "rev_ctrl")
        print(hdr)
        for target_b in benchmarks:
            row_vals = {}
            for cond in conditions:
                p = cache_path(target_b, cond)
                if not p.exists():
                    continue
                acts = load_acts(p)
                r = apply_probe(probe, acts)
                if r is not None:
                    row_vals[cond] = r[0]
            cells = []
            for cond in conditions:
                v = row_vals.get(cond)
                cells.append(("%10.3f" % v) if v is not None else " " * 10)
            print("%-12s %s" % (target_b, " ".join(cells)))
        print()

    # Summary
    print()
    print("=" * 95)
    print("BANK SPECIFICITY SUMMARY: v3 Δ vs rev_control Δ")
    print("(ratio > 1 = v3 more bank-specific than random rewrite)")
    print("-" * 95)
    print("%-12s %-12s %12s %12s %12s" % ("probe", "target", "v3 D", "ctrl D", "spec_ratio"))
    for probe_b in probes:
        probe = probes[probe_b]
        for target_b in benchmarks:
            try:
                oe_p = cache_path(target_b, "orig_eval")
                v3_p = cache_path(target_b, "v3_removed")
                ct_p = cache_path(target_b, "rev_control")
                if not (oe_p.exists() and v3_p.exists() and ct_p.exists()):
                    continue
                oe = apply_probe(probe, load_acts(oe_p))
                v3 = apply_probe(probe, load_acts(v3_p))
                ct = apply_probe(probe, load_acts(ct_p))
                if not all((oe, v3, ct)):
                    continue
                v3d = v3[0] - oe[0]
                cd = ct[0] - oe[0]
                ratio_str = ("%+.2fx" % (v3d / cd)) if cd != 0 else "n/a"
                print("%-12s %-12s %+12.3f %+12.3f %12s" % (probe_b, target_b, v3d, cd, ratio_str))
            except Exception:
                pass


if __name__ == "__main__":
    main()
