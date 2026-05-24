"""Apply trained probes to all conditions, saving per-sample P(eval) predictions.

Usage:
    python -m activation_probing.apply_probes --benchmark hawthorne
    python -m activation_probing.apply_probes --all
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.extract_activations import cache_path


def _load_acts(path: Path) -> tuple[dict[int, np.ndarray], list[str]]:
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    return ({li: z[f"layer_{li}"] for li in layer_indices}, list(z["sample_ids"]))


def apply_for_benchmark(benchmark: str) -> dict | None:
    probe_path = config.PROBES_DIR / f"{benchmark}__probes.pkl"
    if not probe_path.exists():
        print(f"  [missing probe] {probe_path}")
        return None

    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    best_layer = probe["best_layer"]
    scaler = probe["scaler"]
    clf = probe["classifier"]
    print(f"\n=== Applying probe ({benchmark}, layer {best_layer}) ===")

    per_condition = {}
    for cond in config.CONDITIONS:
        path = cache_path(benchmark, cond)
        if not path.exists():
            print(f"  [skip] no cache for {cond}")
            continue
        acts, sample_ids = _load_acts(path)
        if best_layer not in acts:
            print(f"  [skip] layer {best_layer} not cached for {cond}")
            continue
        X = acts[best_layer]
        Xs = scaler.transform(X)
        proba = clf.predict_proba(Xs)[:, 1]  # P(eval)
        per_condition[cond] = {
            "n": int(len(proba)),
            "mean_p_eval": float(proba.mean()),
            "std_p_eval": float(proba.std()),
            "median_p_eval": float(np.median(proba)),
            "predictions": [
                {"sample_id": sid, "p_eval": float(p)} for sid, p in zip(sample_ids, proba)
            ],
        }
        print(f"  {cond:<14}: n={len(proba)}, mean P(eval)={proba.mean():.3f}")

    out = {
        "benchmark": benchmark,
        "model": config.MODEL_NAME,
        "best_layer": int(best_layer),
        "best_layer_cv_acc": float(probe["best_layer_cv_acc"]),
        "per_condition": per_condition,
    }
    out_path = config.RESULTS_DIR / f"{benchmark}__predictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  [saved] {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or not args.benchmark:
        for b in config.BENCHMARKS:
            try:
                apply_for_benchmark(b)
            except Exception as exc:
                print(f"  ERR applying {b}: {exc}")
        return

    apply_for_benchmark(args.benchmark)


if __name__ == "__main__":
    main()
