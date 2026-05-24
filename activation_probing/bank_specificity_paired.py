"""Compute bank-specificity ratio on PAIRED sample IDs only.

Previous run compared orig_eval (n=150) to v3_removed (n=17) to rev_control (n=16) —
mismatched n, baseline computed on full E set rather than paired subset. This recomputes
baseline only on the E-case IDs that have BOTH a corresponding v3_removed AND a
corresponding rev_control entry.

For each benchmark, output:
- N paired
- P(eval) on orig_eval[paired]
- P(eval) on ablated[paired]
- P(eval) on rev_control[paired]
- raw Δ, control Δ, ratio
- Bootstrap CI on raw Δ and ratio
"""
from __future__ import annotations
import json
import pickle
import re
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
    return {int(li): z[f"layer_{li}"] for li in list(z["layer_indices"])}, list(z["sample_ids"])


def normalize_id(sid):
    """Strip __removed, __rev_ctrl, __hyp suffixes etc to get the base sample ID."""
    s = sid
    for suf in ("__rev_ctrl", "__heldout_removed", "__removed", "__hyp"):
        if suf in s:
            s = s.split(suf)[0]
    return s


def bootstrap_ci(values, n_iter=1000, ci=0.95):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    means = np.empty(n_iter)
    for i in range(n_iter):
        means[i] = values[rng.integers(0, len(values), size=len(values))].mean()
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return (lo, hi)


def train_pooled_probe(layer):
    pos_list, neg_list = [], []
    for b in ["needham", "hawthorne", "mask"]:
        oe_p = cache_path(b, "orig_eval")
        od_p = cache_path(b, "orig_deploy")
        if not (oe_p.exists() and od_p.exists()):
            continue
        oe, _ = load_acts(oe_p)
        od, _ = load_acts(od_p)
        if layer not in oe or layer not in od:
            continue
        pos_list.append(oe[layer])
        neg_list.append(od[layer])
    pos = np.concatenate(pos_list, axis=0)
    neg = np.concatenate(neg_list, axis=0)
    X = np.concatenate([pos, neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    cv = cross_val_score(clf_cv, Xs, y, cv=skf, scoring="accuracy").mean()
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(Xs, y)
    return scaler, clf, float(cv)


def predict_p_eval_by_id(probe, acts_map_layer, sample_ids):
    scaler, clf, _ = probe
    X = scaler.transform(acts_map_layer)
    proba = clf.predict_proba(X)[:, 1]
    return {sid: float(p) for sid, p in zip(sample_ids, proba)}


def main():
    print("Model:", config.MODEL_NAME)
    layer = 12  # pooled probe best layer
    probe = train_pooled_probe(layer)
    print(f"Pooled probe at layer {layer}, CV {probe[2]:.3f}")
    print()

    print("=" * 100)
    print("PAIRED bank-specificity (orig vs ablation vs control on SAME sample IDs)")
    print("-" * 100)

    rows = []
    for b, cond_ablation in [("hawthorne", "v3_removed"), ("sad", "v2_removed")]:
        oe_p = cache_path(b, "orig_eval")
        ab_p = cache_path(b, cond_ablation)
        rc_p = cache_path(b, "rev_control")
        if not all(p.exists() for p in (oe_p, ab_p, rc_p)):
            print(f"{b}: missing files, skipping")
            continue
        oe_acts, oe_ids = load_acts(oe_p)
        ab_acts, ab_ids = load_acts(ab_p)
        rc_acts, rc_ids = load_acts(rc_p)
        if layer not in oe_acts or layer not in ab_acts or layer not in rc_acts:
            print(f"{b}: layer {layer} not cached, skipping")
            continue

        # Build per-id P(eval) maps using ORIGINAL (raw) sample IDs
        oe_pred_raw = predict_p_eval_by_id(probe, oe_acts[layer], oe_ids)
        ab_pred_raw = predict_p_eval_by_id(probe, ab_acts[layer], ab_ids)
        rc_pred_raw = predict_p_eval_by_id(probe, rc_acts[layer], rc_ids)

        # Normalize ALL three to base IDs (raw_id -> base_id, and base_id -> p_eval)
        oe_base_to_p = {normalize_id(sid): oe_pred_raw[sid] for sid in oe_ids}
        ab_base_to_p = {normalize_id(sid): ab_pred_raw[sid] for sid in ab_ids}
        rc_base_to_p = {normalize_id(sid): rc_pred_raw[sid] for sid in rc_ids}

        paired_bases = set(oe_base_to_p.keys()) & set(ab_base_to_p.keys()) & set(rc_base_to_p.keys())
        if not paired_bases:
            print(f"{b}: zero paired samples after normalization")
            print(f"  oe sample IDs: {oe_ids[:3]}")
            print(f"  ab sample IDs: {ab_ids[:3]}")
            print(f"  rc sample IDs: {rc_ids[:3]}")
            continue

        per_id_deltas = []
        oe_ps = []
        ab_ps = []
        rc_ps = []
        for base in sorted(paired_bases):
            oe_ps.append(oe_base_to_p[base])
            ab_ps.append(ab_base_to_p[base])
            rc_ps.append(rc_base_to_p[base])
            per_id_deltas.append({
                "id": base,
                "oe_p": oe_base_to_p[base],
                "ab_p": ab_base_to_p[base],
                "rc_p": rc_base_to_p[base],
                "raw_delta": ab_base_to_p[base] - oe_base_to_p[base],
                "ctrl_delta": rc_base_to_p[base] - oe_base_to_p[base],
            })

        n_paired = len(oe_ps)
        if n_paired < 3:
            print(f"{b}: only {n_paired} paired samples, too few")
            continue

        oe_ps = np.array(oe_ps)
        ab_ps = np.array(ab_ps)
        rc_ps = np.array(rc_ps)

        oe_mean = float(oe_ps.mean())
        ab_mean = float(ab_ps.mean())
        rc_mean = float(rc_ps.mean())
        raw_delta = ab_mean - oe_mean
        ctrl_delta = rc_mean - oe_mean
        ratio = raw_delta / ctrl_delta if ctrl_delta != 0 else float("nan")

        # Bootstrap CIs on per-sample deltas
        raw_d_samples = ab_ps - oe_ps
        ctrl_d_samples = rc_ps - oe_ps
        raw_ci = bootstrap_ci(raw_d_samples)
        ctrl_ci = bootstrap_ci(ctrl_d_samples)
        # Causal effect (per-id paired): bank_delta - ctrl_delta
        causal_samples = raw_d_samples - ctrl_d_samples
        causal_mean = float(causal_samples.mean())
        causal_ci = bootstrap_ci(causal_samples)

        print(f"\n{b.upper()} (paired n={n_paired}, ablation={cond_ablation}):")
        print(f"  P(eval) on orig (paired):    {oe_mean:.3f}")
        print(f"  P(eval) on ablated:          {ab_mean:.3f}")
        print(f"  P(eval) on rev_control:      {rc_mean:.3f}")
        print(f"  raw Δ (bank):                {raw_delta:+.3f}  95% CI [{raw_ci[0]:+.3f}, {raw_ci[1]:+.3f}]")
        print(f"  ctrl Δ (matched rewrite):    {ctrl_delta:+.3f}  95% CI [{ctrl_ci[0]:+.3f}, {ctrl_ci[1]:+.3f}]")
        print(f"  CAUSAL Δ (bank - ctrl):      {causal_mean:+.3f}  95% CI [{causal_ci[0]:+.3f}, {causal_ci[1]:+.3f}]")
        print(f"  bank-spec ratio raw_Δ/ctrl_Δ: {ratio:.2f}x")

        rows.append({
            "benchmark": b,
            "ablation": cond_ablation,
            "n_paired": n_paired,
            "oe_mean_paired": oe_mean,
            "ab_mean_paired": ab_mean,
            "rc_mean_paired": rc_mean,
            "raw_delta": raw_delta,
            "raw_delta_ci95": list(raw_ci),
            "ctrl_delta": ctrl_delta,
            "ctrl_delta_ci95": list(ctrl_ci),
            "causal_delta": causal_mean,
            "causal_delta_ci95": list(causal_ci),
            "bank_spec_ratio": ratio,
        })

    out = config.RESULTS_DIR / "bank_specificity_paired.json"
    out.write_text(json.dumps({"model": config.MODEL_NAME, "layer": layer,
                               "pooled_cv": probe[2], "results": rows}, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
