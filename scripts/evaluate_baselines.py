"""
Run Oracle and Random sanity baselines on the test set.

Demonstrates that the evaluation framework works correctly:
  1. ORACLE baseline: Assigns score = ground_truth_label.
     Expected output: 1.0000 across Accuracy, Precision, Recall, F1, and ROC AUC.
  2. RANDOM baseline: Assigns uniform random scores in [0, 1] with a fixed seed.
     Expected output: Chance-level performance (~0.50 ROC AUC).

Outputs:
  - results/baseline_oracle_preds.csv & results/baseline_oracle.json
  - results/baseline_random_preds.csv & results/baseline_random.json
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.evaluation import (
    Timer,
    best_f1_threshold,
    compute_metrics,
    per_error_type_recall,
    save_results
)


def run_baselines(testset_path: str, output_dir: str, seed: int = 42) -> None:
    if not os.path.exists(testset_path):
        print(f"Error: Test set not found at {testset_path}")
        print("Please run: python scripts/build_testset.py first.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    df_test = pd.read_csv(testset_path)
    y_true = df_test["label"].astype(int).values
    n_rows = len(df_test)

    print("=" * 70)
    print(f"EVALUATING SANITY BASELINES ON TEST SET ({n_rows} sentences)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. ORACLE BASELINE: Score = True Label
    # -------------------------------------------------------------
    with Timer("oracle_baseline") as t_oracle:
        oracle_scores = y_true.astype(float)

    df_oracle_preds = pd.DataFrame({
        "row_id": df_test["row_id"],
        "hallucination_score": oracle_scores,
        "seconds": round(t_oracle.elapsed_seconds / max(n_rows, 1), 6)
    })
    oracle_preds_path = os.path.join(output_dir, "baseline_oracle_preds.csv")
    df_oracle_preds.to_csv(oracle_preds_path, index=False)

    oracle_metrics = compute_metrics(y_true, oracle_scores, threshold=0.5)
    oracle_meta = {
        "approach": "Baseline",
        "model_name": "Oracle (Ground Truth)",
        "testset_path": testset_path,
        "seconds_total": t_oracle.elapsed_seconds,
        "seconds_per_sentence": round(t_oracle.elapsed_seconds / max(n_rows, 1), 6)
    }
    oracle_json_path = os.path.join(output_dir, "baseline_oracle.json")
    save_results(oracle_metrics, oracle_json_path, metadata=oracle_meta)

    # -------------------------------------------------------------
    # 2. RANDOM BASELINE: Uniform random scores [0, 1]
    # -------------------------------------------------------------
    rng = np.random.RandomState(seed)
    with Timer("random_baseline") as t_random:
        random_scores = rng.uniform(0.0, 1.0, size=n_rows)

    df_random_preds = pd.DataFrame({
        "row_id": df_test["row_id"],
        "hallucination_score": random_scores.round(4),
        "seconds": round(t_random.elapsed_seconds / max(n_rows, 1), 6)
    })
    random_preds_path = os.path.join(output_dir, "baseline_random_preds.csv")
    df_random_preds.to_csv(random_preds_path, index=False)

    random_metrics = compute_metrics(y_true, random_scores, threshold=0.5)
    random_optimistic = best_f1_threshold(y_true, random_scores)
    random_meta = {
        "approach": "Baseline",
        "model_name": f"Random Scorer (Uniform, seed={seed})",
        "testset_path": testset_path,
        "optimistic_best_f1": random_optimistic,
        "seconds_total": t_random.elapsed_seconds,
        "seconds_per_sentence": round(t_random.elapsed_seconds / max(n_rows, 1), 6)
    }
    random_json_path = os.path.join(output_dir, "baseline_random.json")
    save_results(random_metrics, random_json_path, metadata=random_meta)

    # -------------------------------------------------------------
    # COMPARISON PRINT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS (Threshold = 0.5)")
    print("=" * 70)
    header = f"{'Metric':<25} | {'Oracle':<15} | {'Random (seed=42)':<15}"
    print(header)
    print("-" * len(header))
    print(f"{'Accuracy':<25} | {oracle_metrics['accuracy']:<15.4f} | {random_metrics['accuracy']:<15.4f}")
    print(f"{'Precision':<25} | {oracle_metrics['precision']:<15.4f} | {random_metrics['precision']:<15.4f}")
    print(f"{'Recall':<25} | {oracle_metrics['recall']:<15.4f} | {random_metrics['recall']:<15.4f}")
    print(f"{'F1 Score':<25} | {oracle_metrics['f1']:<15.4f} | {random_metrics['f1']:<15.4f}")
    print(f"{'ROC AUC':<25} | {oracle_metrics['roc_auc']:<15.4f} | {random_metrics['roc_auc']:<15.4f}")
    print(f"{'Average Precision':<25} | {oracle_metrics['average_precision']:<15.4f} | {random_metrics['average_precision']:<15.4f}")
    print("=" * 70)
    print(f"Oracle Confusion Matrix: {oracle_metrics['confusion_matrix']}")
    print(f"Random Confusion Matrix: {random_metrics['confusion_matrix']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Oracle and Random baselines.")
    parser.add_argument(
        "--testset",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to testset.csv"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "results"),
        help="Directory to save baseline predictions and evaluation results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random baseline")
    args = parser.parse_args()

    run_baselines(
        testset_path=os.path.abspath(args.testset),
        output_dir=os.path.abspath(args.output_dir),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
