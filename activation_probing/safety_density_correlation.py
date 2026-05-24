"""(3) Safety-content density correlation.

For each test transcript, compute a safety-content score (refusal/harm keyword density).
Correlate with the probe's P(eval). If high correlation → probe is partly a safety
detector. If no correlation → probe is doing something else.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path
from activation_probing.data_loader import load_condition

import os
PROBES_DIR_OVERRIDE = os.environ.get("PROBES_DIR")
if PROBES_DIR_OVERRIDE:
    config.PROBES_DIR = Path(PROBES_DIR_OVERRIDE)
RESULTS_DIR_OVERRIDE = os.environ.get("RESULTS_DIR")
if RESULTS_DIR_OVERRIDE:
    config.RESULTS_DIR = Path(RESULTS_DIR_OVERRIDE)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SAFETY_KEYWORDS = [
    "refus", "harm", "danger", "unsafe", "policy", "guideline", "violat",
    "ethic", "illegal", "weapon", "explos", "drug", "abuse", "exploit",
    "malicious", "fraud", "scam", "manipulat", "deceiv", "decept", "lie",
    "phish", "hack", "attack", "vulnerab", "credential", "password",
    "kill", "murder", "suicid", "self-harm", "child", "minor", "porn",
    "racist", "discriminat", "hate", "bias", "stereo", "propagand",
    "disinform", "misinform", "fake news",
]


def transcript_text(msgs):
    parts = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
        parts.append(str(c))
    return " ".join(parts)


def safety_density(text):
    text_l = text.lower()
    words = text_l.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(kw in w for kw in SAFETY_KEYWORDS))
    return hits / len(words)


def load_acts(path):
    z = np.load(path, allow_pickle=False)
    return {int(li): z[f"layer_{li}"] for li in list(z["layer_indices"])}, list(z["sample_ids"])


def main():
    print("Model:", config.MODEL_NAME)
    # Use pooled probe results if they exist; else use per-benchmark probes
    probes = {}
    for b in ["needham", "hawthorne", "mask"]:
        p = config.PROBES_DIR / f"{b}__probes.pkl"
        if p.exists():
            with open(p, "rb") as f:
                probes[b] = pickle.load(f)

    # For each (probe_benchmark, target_benchmark), compute safety_density vs P(eval)
    all_rows = []
    for probe_b, probe in probes.items():
        layer = int(probe["best_layer"])
        scaler = probe["scaler"]
        clf = probe["classifier"]
        for target_b in ["needham", "hawthorne", "mask"]:
            cache_p = cache_path(target_b, "orig_eval")
            if not cache_p.exists():
                continue
            acts, sample_ids = load_acts(cache_p)
            if layer not in acts:
                continue
            X = acts[layer]
            Xs = scaler.transform(X)
            p_evals = clf.predict_proba(Xs)[:, 1]
            # Now compute safety density per transcript
            items = load_condition(target_b, "orig_eval")
            sid_to_msgs = {sid: msgs for sid, msgs in items}
            for sid, p_eval in zip(sample_ids, p_evals):
                msgs = sid_to_msgs.get(sid)
                if not msgs:
                    continue
                text = transcript_text(msgs)
                dens = safety_density(text)
                all_rows.append({
                    "probe": probe_b,
                    "target": target_b,
                    "sid": sid,
                    "p_eval": float(p_eval),
                    "safety_density": float(dens),
                })

    print(f"Collected {len(all_rows)} (sample, probe, p_eval, safety_density) tuples")

    # Aggregate correlation per probe
    print()
    print("=" * 80)
    print("(3) Safety-density vs P(eval) correlation")
    print("-" * 80)
    print(f"{'probe':<15} {'target':<15} {'n':>5} {'Pearson r':>10} {'Spearman r':>12}")
    by_pt = {}
    for r in all_rows:
        key = (r["probe"], r["target"])
        by_pt.setdefault(key, []).append(r)
    for (probe_b, target_b), rs in sorted(by_pt.items()):
        if len(rs) < 5:
            continue
        ps = np.array([r["p_eval"] for r in rs])
        ds = np.array([r["safety_density"] for r in rs])
        if ds.std() < 1e-8:
            print(f"{probe_b:<15} {target_b:<15} {len(rs):>5}  zero variance in safety density")
            continue
        pr = pearsonr(ds, ps).statistic
        sr = spearmanr(ds, ps).statistic
        print(f"{probe_b:<15} {target_b:<15} {len(rs):>5} {pr:>+10.3f} {sr:>+12.3f}")

    # Pooled across all data
    print()
    all_ps = np.array([r["p_eval"] for r in all_rows])
    all_ds = np.array([r["safety_density"] for r in all_rows])
    if all_ds.std() > 1e-8:
        pr = pearsonr(all_ds, all_ps).statistic
        sr = spearmanr(all_ds, all_ps).statistic
        print(f"Pooled across all rows ({len(all_rows)} samples):")
        print(f"  Pearson:  {pr:+.3f}")
        print(f"  Spearman: {sr:+.3f}")
        print()
        if abs(pr) > 0.4:
            print(f"VERDICT: Probe IS partly a safety detector.")
            print(f"  Pearson r = {pr:.3f} between safety-content density and P(eval).")
        elif abs(pr) > 0.2:
            print(f"VERDICT: Probe shows MILD safety-content sensitivity (r = {pr:.3f}).")
        else:
            print(f"VERDICT: Probe is NOT primarily a safety detector (r = {pr:.3f}).")

    out = config.RESULTS_DIR / "safety_density_correlation.json"
    out.write_text(json.dumps({
        "model": config.MODEL_NAME,
        "n_total": len(all_rows),
        "by_probe_target": {f"{k[0]}__{k[1]}": [{"sid": r["sid"], "p_eval": r["p_eval"],
                                                  "safety_density": r["safety_density"]} for r in v]
                            for k, v in by_pt.items()},
    }, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
