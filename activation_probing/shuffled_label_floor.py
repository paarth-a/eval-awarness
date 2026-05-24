"""S2: Shuffled-label probe floor.

Compute null distribution for cross-benchmark probe transfer AUC.
- For each (source, target) pair (off-diagonal):
  - Train probe on source's E vs D with SHUFFLED labels (preserves class balance)
  - Apply to target's true E vs D, compute AUC
  - Repeat N_PERM times
- Compare actual M1 off-diag AUC to the null distribution.

Output: per-pair null mean + 95% CI + p-value of actual AUC.
"""
from __future__ import annotations
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path

PROBES_DIR_OVERRIDE = os.environ.get("PROBES_DIR")
if PROBES_DIR_OVERRIDE:
    config.PROBES_DIR = Path(PROBES_DIR_OVERRIDE)

N_PERM = int(os.environ.get("N_PERM", "100"))


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    return {int(li): z[f"layer_{li}"] for li in list(z["layer_indices"])}


def train_on_shuffled_and_eval(X_src, y_src_shuffled, X_tgt, y_tgt, layer):
    """Train probe on source with shuffled labels, eval on target's true labels."""
    scaler = StandardScaler()
    Xs_src = scaler.fit_transform(X_src)
    Xs_tgt = scaler.transform(X_tgt)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(Xs_src, y_src_shuffled)
    proba = clf.predict_proba(Xs_tgt)[:, 1]
    try:
        return float(roc_auc_score(y_tgt, proba))
    except Exception:
        return float("nan")


def main():
    print(f"Model: {config.MODEL_NAME}")
    print(f"N_PERM: {N_PERM}")

    bms = ["needham", "hawthorne", "mask"]
    all_acts = {}
    for b in bms:
        oe = cache_path(b, "orig_eval")
        od = cache_path(b, "orig_deploy")
        if not (oe.exists() and od.exists()):
            print(f"[skip] {b}: missing acts")
            continue
        all_acts[b] = {"oe": load_acts(oe), "od": load_acts(od)}
    print(f"Loaded acts for: {list(all_acts.keys())}")

    # Load probes for actual M1 AUCs (best_layer per benchmark)
    probes = {}
    for b in bms:
        p = config.PROBES_DIR / f"{b}__probes.pkl"
        if p.exists():
            with open(p, "rb") as f:
                probes[b] = pickle.load(f)
    print(f"Probes loaded: {list(probes.keys())}")

    rng = np.random.default_rng(42)

    # Per (source_probe, target) pair: compute actual AUC + null distribution
    print()
    print("=" * 90)
    print(f"Cross-bench probe transfer — actual AUC vs shuffled-label null (n_perm={N_PERM})")
    print("=" * 90)
    print(f"{'source':<12} {'target':<12} {'best_layer':>10} {'actual AUC':>11} {'null mean':>11} {'null 95% CI':>22} {'percentile':>11}")

    rows = []
    for src_b, probe in probes.items():
        if src_b not in all_acts:
            continue
        best_layer = int(probe["best_layer"])
        for tgt_b in bms:
            if tgt_b not in all_acts:
                continue
            X_src = np.concatenate([all_acts[src_b]["oe"][best_layer], all_acts[src_b]["od"][best_layer]], axis=0).astype(np.float32)
            y_src_true = np.concatenate([np.ones(len(all_acts[src_b]["oe"][best_layer])), np.zeros(len(all_acts[src_b]["od"][best_layer]))])
            X_tgt = np.concatenate([all_acts[tgt_b]["oe"][best_layer], all_acts[tgt_b]["od"][best_layer]], axis=0).astype(np.float32)
            y_tgt = np.concatenate([np.ones(len(all_acts[tgt_b]["oe"][best_layer])), np.zeros(len(all_acts[tgt_b]["od"][best_layer]))])

            # Actual: use the trained source probe directly
            scaler = probe["scaler"]
            clf = probe["classifier"]
            try:
                Xs_tgt = scaler.transform(X_tgt)
                proba = clf.predict_proba(Xs_tgt)[:, 1]
                actual_auc = float(roc_auc_score(y_tgt, proba))
            except Exception:
                actual_auc = float("nan")

            # Null: shuffled labels on source, repeated
            null_aucs = []
            for _ in range(N_PERM):
                idx = rng.permutation(len(y_src_true))
                y_shuf = y_src_true[idx]
                auc = train_on_shuffled_and_eval(X_src, y_shuf, X_tgt, y_tgt, best_layer)
                if not np.isnan(auc):
                    null_aucs.append(auc)
            null = np.array(null_aucs)
            null_mean = float(null.mean()) if len(null) else float("nan")
            null_lo = float(np.percentile(null, 2.5)) if len(null) else float("nan")
            null_hi = float(np.percentile(null, 97.5)) if len(null) else float("nan")
            pct = float((actual_auc > null).mean()) if len(null) else float("nan")

            label = "[SELF]" if src_b == tgt_b else ""
            print(f"{src_b:<12} {tgt_b:<12} {best_layer:>10} {actual_auc:>11.3f} {null_mean:>11.3f} [{null_lo:>+7.3f}, {null_hi:>+7.3f}] {pct:>11.3f} {label}")

            rows.append({
                "source": src_b, "target": tgt_b, "best_layer": best_layer,
                "actual_auc": actual_auc, "null_mean": null_mean,
                "null_ci95": [null_lo, null_hi], "percentile_of_actual": pct,
                "is_self": src_b == tgt_b,
            })

    # Aggregate off-diagonal
    off = [r for r in rows if not r["is_self"]]
    if off:
        actual_mean = np.mean([r["actual_auc"] for r in off])
        null_grand_mean = np.mean([r["null_mean"] for r in off])
        # Pool all null AUCs across off-diag pairs
        print()
        print("=" * 70)
        print("OFF-DIAGONAL POOLED")
        print(f"  Mean actual off-diag AUC:  {actual_mean:.3f}")
        print(f"  Mean null off-diag AUC:    {null_grand_mean:.3f}")
        print(f"  Difference (actual−null):  {actual_mean - null_grand_mean:+.3f}")

    out = {
        "model": config.MODEL_NAME,
        "n_perm": N_PERM,
        "per_pair": rows,
        "mean_actual_off_diag_auc": float(actual_mean) if off else None,
        "mean_null_off_diag_auc": float(null_grand_mean) if off else None,
    }
    out_path = config.RESULTS_DIR / "shuffled_label_floor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
