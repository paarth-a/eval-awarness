"""Run all analysis for a specified model and output results.

Usage:
    python run_analysis.py --model gpt-5-mini
    python run_analysis.py --model gpt-5-mini --data-file eval_awareness_data/dataset.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score, brier_score_loss


def main():
    parser = argparse.ArgumentParser(description="Run full analysis for a model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., gpt-5-mini)",
    )
    parser.add_argument(
        "--data-file", type=str, default="eval_awareness_data/dataset.json",
        help="Path to dataset.json",
    )
    args = parser.parse_args()

    model = args.model
    os.environ["EVAL_AWARENESS_DATA_FILE"] = args.data_file
    os.environ["MODELS_TO_PLOT"] = model

    from analysis.find_data import find_dataset
    from analysis.score_table import expected_calibration_error
    from inspect_ai.log import read_eval_log
    from constants import find_most_recent_file

    print("=" * 60)
    print(f"{model} — Full Analysis Report")
    print("=" * 60)

    # --- 1. Score Table (AUC / Brier / ECE) ---
    print("\n--- Eval Detection Scores (eval_mcq) ---")
    print(f"{'Subset':<10} {'AUC':>6} {'Brier':>6} {'ECE':>6}")
    print("-" * 32)

    data = find_dataset(model, experiment_dir="eval_mcq")
    if data is None:
        print("ERROR: No eval_mcq data found!")
        sys.exit(1)

    results = {}
    for subset_name, agent_filter in [("All", None), ("Chat", 0), ("Agent", 1)]:
        subset = data if agent_filter is None else data[data["agent"] == agent_filter]
        grouped = (
            subset[["sample_id", "cat_eval", "human", "score"]]
            .groupby("sample_id")
            .mean()
            .reset_index()
        )
        auc = roc_auc_score(grouped["cat_eval"], grouped["score"])
        brier = brier_score_loss(grouped["cat_eval"], grouped["score"])
        ece = expected_calibration_error(
            grouped["cat_eval"], grouped["score"], n_bins=11, strategy="uniform"
        )
        results[subset_name] = {"AUC": auc, "Brier": brier, "ECE": ece}
        print(f"{subset_name:<10} {auc:>6.3f} {brier:>6.3f} {ece:>6.3f}")

    print(f"\nTotal samples: {len(data)}, Unique transcripts: {data['sample_id'].nunique()}")

    # --- 2. Purpose MCQ ---
    print("\n--- Purpose Detection (purpose_mcq) ---")
    purpose_data = find_dataset(model, experiment_dir="purpose_mcq")
    if purpose_data is not None:
        for subset_name, agent_filter in [("All", None), ("Chat", 0), ("Agent", 1)]:
            subset = purpose_data if agent_filter is None else purpose_data[purpose_data["agent"] == agent_filter]
            grouped = (
                subset[["sample_id", "cat_eval", "score"]]
                .groupby("sample_id")
                .mean()
                .reset_index()
            )
            print(f"  {subset_name}: {len(subset)} rows, {subset['sample_id'].nunique()} unique")
    else:
        print("  No purpose_mcq data found")

    # --- 3. CoT Classification (eval_mcq_cot) ---
    print("\n--- CoT Classification (eval_mcq_cot) ---")
    cot_file = find_most_recent_file(f"logs/eval_mcq_cot/{model}")
    if cot_file:
        cot_log = read_eval_log(cot_file, format="eval")
        print(f"  Samples: {len(cot_log.samples)}")
        print(f"  Accuracy: {cot_log.results.scores[0].metrics['accuracy'].value:.3f}")

        label_counts = Counter()
        for sample in cot_log.samples:
            for score_result in sample.scores.values():
                if hasattr(score_result, "metadata") and score_result.metadata:
                    label = score_result.metadata.get("label", "unknown")
                    label_counts[label] += 1
        if label_counts:
            print("  Label distribution:")
            for label, count in label_counts.most_common():
                print(f"    {label}: {count}")
    else:
        print("  No eval_mcq_cot data found")

    # --- 4. Purpose Open CoT (purpose_open_cot) ---
    print("\n--- Purpose Open-Ended CoT (purpose_open_cot) ---")
    poc_file = find_most_recent_file(f"logs/purpose_open_cot/{model}")
    if poc_file:
        poc_log = read_eval_log(poc_file, format="eval")
        print(f"  Samples: {len(poc_log.samples)}")
        print(f"  Accuracy: {poc_log.results.scores[0].metrics['accuracy'].value:.3f}")
    else:
        print("  No purpose_open_cot data found")

    # --- 5. Save JSON results ---
    output = {
        "model": model,
        "eval_detection": {
            subset: {k: round(v, 4) for k, v in metrics.items()}
            for subset, metrics in results.items()
        },
    }
    output_dir = Path(f"results/{model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scores.json"
    output_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
