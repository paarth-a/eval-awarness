"""Train per-layer logistic-regression probes on cached activations.

For each benchmark, load orig_eval and orig_deploy activations, train a probe
at each cached layer, pick the best-layer probe by held-out accuracy, save it.

Usage:
    python -m activation_probing.train_probes --benchmark hawthorne
    python -m activation_probing.train_probes --all
"""
from __future__ import annotations

import argparse
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


def _load_npz(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    layer_indices = list(z["layer_indices"])
    activations = {li: z[f"layer_{li}"] for li in layer_indices}
    sample_ids = z["sample_ids"]
    return activations, sample_ids


def train_for_benchmark(benchmark: str, overwrite: bool = False) -> dict | None:
    out_path = config.PROBES_DIR / f"{benchmark}__probes.pkl"
    summary_path = config.RESULTS_DIR / f"{benchmark}__probe_training.json"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name}")
        return None

    eval_path = cache_path(benchmark, "orig_eval")
    deploy_path = cache_path(benchmark, "orig_deploy")
    if not (eval_path.exists() and deploy_path.exists()):
        print(f"  [missing] {benchmark}: need both orig_eval and orig_deploy caches")
        return None

    print(f"\n=== Training probes for {benchmark} ===")
    eval_acts, _ = _load_npz(eval_path)
    deploy_acts, _ = _load_npz(deploy_path)

    layer_indices = sorted(set(eval_acts.keys()) & set(deploy_acts.keys()))
    if not layer_indices:
        print(f"  [error] no shared layers between orig_eval and orig_deploy")
        return None

    skf = StratifiedKFold(n_splits=config.PROBE_CV_FOLDS, shuffle=True, random_state=config.RNG_SEED)

    per_layer = {}
    for li in layer_indices:
        X = np.concatenate([eval_acts[li], deploy_acts[li]], axis=0)
        y = np.concatenate([
            np.ones(len(eval_acts[li]), dtype=np.int32),
            np.zeros(len(deploy_acts[li]), dtype=np.int32),
        ])
        if X.shape[0] < 2 * config.PROBE_CV_FOLDS:
            print(f"  [skip layer {li}] too few samples ({X.shape[0]})")
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(C=config.PROBE_C, max_iter=2000, random_state=config.RNG_SEED)
        cv_scores = cross_val_score(clf, Xs, y, cv=skf, scoring="accuracy")
        # Train final classifier on full data (use this for inference)
        clf_full = LogisticRegression(C=config.PROBE_C, max_iter=2000, random_state=config.RNG_SEED)
        clf_full.fit(Xs, y)
        per_layer[li] = {
            "cv_acc_mean": float(cv_scores.mean()),
            "cv_acc_std": float(cv_scores.std()),
            "n_eval": int(len(eval_acts[li])),
            "n_deploy": int(len(deploy_acts[li])),
            "scaler": scaler,
            "classifier": clf_full,
        }
        print(f"  layer {li:>3d}: cv_acc = {cv_scores.mean():.3f} ± {cv_scores.std():.3f} "
              f"(n_eval={len(eval_acts[li])}, n_deploy={len(deploy_acts[li])})")

    if not per_layer:
        print(f"  [empty] no layers trained")
        return None

    best_layer = max(per_layer.items(), key=lambda kv: kv[1]["cv_acc_mean"])[0]
    print(f"  best layer: {best_layer} (acc={per_layer[best_layer]['cv_acc_mean']:.3f})")

    # Save the best-layer probe (and also keep all-layer accuracies for reporting)
    payload = {
        "benchmark": benchmark,
        "model": config.MODEL_NAME,
        "best_layer": best_layer,
        "best_layer_cv_acc": per_layer[best_layer]["cv_acc_mean"],
        "scaler": per_layer[best_layer]["scaler"],
        "classifier": per_layer[best_layer]["classifier"],
        "all_layers_acc": {li: v["cv_acc_mean"] for li, v in per_layer.items()},
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    summary_path.write_text(json.dumps({
        "benchmark": benchmark,
        "model": config.MODEL_NAME,
        "best_layer": int(best_layer),
        "best_layer_cv_acc": float(per_layer[best_layer]["cv_acc_mean"]),
        "all_layers_acc": {int(li): float(v["cv_acc_mean"]) for li, v in per_layer.items()},
    }, indent=2))
    print(f"  [saved] {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all or not args.benchmark:
        for b in config.BENCHMARKS:
            try:
                train_for_benchmark(b, overwrite=args.overwrite)
            except Exception as exc:
                print(f"  ERR training {b}: {exc}")
        return

    train_for_benchmark(args.benchmark, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
