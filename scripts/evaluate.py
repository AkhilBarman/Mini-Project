"""
Evaluate hallucination detector predictions against the ground-truth test set.

Usage:
  python scripts/evaluate.py --predictions <path_to_preds.csv>
"""

import argparse
import os
import sys
import pandas as pd

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.evaluation import (
    best_f1_threshold,
    compute_metrics,
    per_error_type_recall,
    save_results
)


def run_evaluation(
    predictions_path: str,
    testset_path: str,
    output_dir: str,
    threshold: float = 0.5
) -> None:
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at {predictions_path}")
        sys.exit(1)

    if not os.path.exists(testset_path):
        print(f"Error: Test set not found at {testset_path}")
        print("Please run: python scripts/build_testset.py first.")
        sys.exit(1)

    df_preds = pd.read_csv(predictions_path)
    df_test = pd.read_csv(testset_path)

    # Validate predictions schema contract
    required_cols = {"row_id", "hallucination_score"}
    if not required_cols.issubset(df_preds.columns):
        print(f"Error: Predictions CSV must have columns: {required_cols}. Found: {df_preds.columns.tolist()}")
        sys.exit(1)

    # Merge on row_id
    merged = pd.merge(df_test, df_preds, on="row_id", how="inner")
    if len(merged) != len(df_test):
        print(f"Warning: Merged {len(merged)} rows out of {len(df_test)} test set rows.")

    y_true = merged["label"].astype(int).values
    scores = merged["hallucination_score"].astype(float).values

    # Compute metrics
    metrics = compute_metrics(y_true, scores, threshold=threshold)
    optimistic = best_f1_threshold(y_true, scores)
    type_recall = per_error_type_recall(merged, scores, threshold=threshold)

    # Print summary report
    print("\n" + "=" * 65)
    print("HALLUCINATION DETECTION EVALUATION REPORT")
    print("=" * 65)
    print(f"Predictions File:          {os.path.abspath(predictions_path)}")
    print(f"Testset File:              {os.path.abspath(testset_path)}")
    print(f"Total Evaluated Sentences: {len(merged)}")
    print(f"Positives (Hallucinated):  {metrics['n_pos']} | Negatives (Supported): {metrics['n_neg']}")
    print("-" * 65)
    print(f"Decision Threshold:        {threshold:.2f}")
    print(f"Accuracy:                  {metrics['accuracy']:.4f}")
    print(f"Precision:                 {metrics['precision']:.4f}")
    print(f"Recall:                    {metrics['recall']:.4f}")
    print(f"F1 Score:                  {metrics['f1']:.4f}")
    print(f"ROC AUC:                   {metrics['roc_auc']:.4f}")
    print(f"Average Precision (PR AUC):{metrics['average_precision']:.4f}")
    print("-" * 65)
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix:          TP: {cm['tp']} | FP: {cm['fp']} | TN: {cm['tn']} | FN: {cm['fn']}")
    print("-" * 65)
    print(f"Optimistic Best Threshold: {optimistic['best_threshold']:.2f} -> F1: {optimistic['best_f1']:.4f}")
    print(f"({optimistic['note']})")
    print("-" * 65)
    print("Recall by Error Type:")
    for etype, data in sorted(type_recall.items()):
        print(f"  - {etype:<15}: {data['detected']}/{data['total']} detected ({data['recall'] * 100:.1f}%)")
    print("=" * 65 + "\n")

    # Save output
    base_name = os.path.splitext(os.path.basename(predictions_path))[0]
    out_json = os.path.join(output_dir, f"{base_name}_eval.json")
    metadata = {
        "predictions_file": os.path.abspath(predictions_path),
        "testset_file": os.path.abspath(testset_path),
        "threshold": threshold,
        "optimistic_best_f1": optimistic,
        "per_error_type_recall": type_recall,
        "seconds_total": float(df_preds["seconds"].sum()) if "seconds" in df_preds.columns else None
    }
    save_results(metrics, out_json, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination detector predictions.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument(
        "--testset",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to testset.csv"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "results"),
        help="Directory to save evaluation JSON results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5)"
    )
    args = parser.parse_args()

    run_evaluation(
        predictions_path=os.path.abspath(args.predictions),
        testset_path=os.path.abspath(args.testset),
        output_dir=os.path.abspath(args.output_dir),
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
