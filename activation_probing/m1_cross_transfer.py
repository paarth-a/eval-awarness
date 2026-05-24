"""M1: cross-benchmark probe transfer matrix.

For each (source_probe, target_benchmark) pair, evaluate the source probe's
ability to classify the TARGET benchmark's orig_eval (label=1) vs orig_deploy
(label=0). Report AUROC.

A high AUC off-diagonal means the probe captures something that generalizes
across benchmarks (not just "looks like benchmark X eval distribution").
A near-chance AUC means the probe is benchmark-bound — the concerning case
flagged by the user.

Usage:
    python -m activation_probing.m1_cross_transfer

Output: results/m1_cross_transfer.json + printed matrix.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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
    layer_indices = list(z["layer_indices"])
    return {int(li): z[f"layer_{li}"] for li in layer_indices}


def main():
    print("Model:", config.MODEL_NAME)
    print()

    probes = {}
    for b in ["needham", "hawthorne", "mask"]:
        p = config.PROBES_DIR / f"{b}__probes.pkl"
        if p.exists():
            with open(p, "rb") as f:
                probes[b] = pickle.load(f)
            print(f"  probe {b}: best_layer={probes[b]['best_layer']} cv_acc={probes[b]['best_layer_cv_acc']:.3f}")
    if not probes:
        print("No probes found at", config.PROBES_DIR)
        return

    # For each (probe, target_b) compute AUC of P(eval) on target's orig_eval vs orig_deploy
    matrix = {}
    targets = ["needham", "hawthorne", "mask"]
    print()
    print("=" * 70)
    print("M1 — Cross-benchmark probe transfer AUC")
    print("Rows = probe trained on; Columns = target benchmark (E vs D)")
    print("-" * 70)
    hdr = "%-12s" % "probe\\target" + "".join("%14s" % t for t in targets)
    print(hdr)

    for probe_b, probe in probes.items():
        layer = int(probe["best_layer"])
        scaler = probe["scaler"]
        clf = probe["classifier"]
        row = {}
        for target_b in targets:
            oe_path = cache_path(target_b, "orig_eval")
            od_path = cache_path(target_b, "orig_deploy")
            if not (oe_path.exists() and od_path.exists()):
                row[target_b] = None
                continue
            oe_acts = load_acts(oe_path)
            od_acts = load_acts(od_path)
            if layer not in oe_acts or layer not in od_acts:
                row[target_b] = None
                continue
            X = np.concatenate([oe_acts[layer], od_acts[layer]], axis=0).astype(np.float32)
            y = np.concatenate([np.ones(len(oe_acts[layer])), np.zeros(len(od_acts[layer]))])
            try:
                Xs = scaler.transform(X)
                proba = clf.predict_proba(Xs)[:, 1]
                auc = float(roc_auc_score(y, proba))
                row[target_b] = auc
            except Exception as e:
                row[target_b] = None
        matrix[probe_b] = row

        cells = ["%-12s" % probe_b]
        for t in targets:
            v = row.get(t)
            cells.append("%14s" % ("%.3f" % v if v is not None else "n/a"))
        print("".join(cells))

    # Mean off-diagonal AUC (probe transfer signal)
    off_diag = []
    diag = []
    for probe_b, row in matrix.items():
        for target_b, auc in row.items():
            if auc is None:
                continue
            if probe_b == target_b:
                diag.append(auc)
            else:
                off_diag.append(auc)
    print()
    print("Diagonal (self) mean AUC:    %.3f (n=%d)" % (np.mean(diag), len(diag)))
    if off_diag:
        print("Off-diagonal (transfer) mean: %.3f (n=%d)" % (np.mean(off_diag), len(off_diag)))
        gap = np.mean(diag) - np.mean(off_diag)
        print("Self - Transfer gap:         %+.3f" % gap)
        print()
        if np.mean(off_diag) > 0.75:
            print("VERDICT: Probe transfer WORKS (off-diag AUC > 0.75).")
            print("  Probe captures generalizable signal, not pure benchmark identity.")
        elif np.mean(off_diag) > 0.6:
            print("VERDICT: Probe transfer PARTIAL (off-diag AUC 0.6-0.75).")
            print("  Probe captures some general signal but also benchmark-specific features.")
        else:
            print("VERDICT: Probe transfer FAILS (off-diag AUC < 0.6).")
            print("  Probe is largely benchmark-bound. Confound concern is serious.")

    out = config.RESULTS_DIR / "m1_cross_transfer.json"
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "matrix": matrix,
        "mean_diag_auc": float(np.mean(diag)) if diag else None,
        "mean_off_diag_auc": float(np.mean(off_diag)) if off_diag else None,
    }, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
