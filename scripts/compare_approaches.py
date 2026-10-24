"""
Compare All Hallucination Detection Approaches.

Evaluates baselines, Approach A, Approach B, and Hybrid Ensemble on the
462-sentence Indian court judgments benchmark (data/testset/testset.csv).
Outputs:
- results/comparison_summary.csv
- results/comparison_summary.md
- results/comparison_summary.json
- results/comparison_breakdown.png
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hd.evaluation import compute_metrics, per_error_type_recall, best_f1_threshold
from hd.comparison import (
    compute_hybrid_scores,
    build_comparison_dataframe,
    build_error_breakdown_dataframe,
    plot_comparison_dashboard,
    dataframe_to_markdown,
)


def evaluate_single_run(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    testset_df: pd.DataFrame,
    runtime_seconds: float = 0.0
) -> dict:
    """Evaluate a single model run with given threshold."""
    metrics = compute_metrics(y_true, scores, threshold=threshold)
    err_breakdown = per_error_type_recall(
        testset_df,
        scores,
        threshold=threshold
    )
    best_opt = best_f1_threshold(y_true, scores)
    
    return {
        "name": name,
        "metrics": metrics,
        "metadata": {
            "per_error_type_recall": err_breakdown,
            "seconds_total": runtime_seconds,
            "optimistic_best_f1": best_opt
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compare hallucination detection approaches.")
    parser.add_argument("--testset", default="data/testset/testset.csv", help="Path to testset.csv")
    parser.add_argument("--results-dir", default="results", help="Directory where predictions reside")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate comparison charts")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 5: COMPARATIVE BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load testset
    if not os.path.exists(args.testset):
        print(f"Error: testset file not found at {args.testset}")
        sys.exit(1)
    
    testset_df = pd.read_csv(args.testset)
    y_true = testset_df["label"].values
    print(f"Loaded benchmark testset: {len(testset_df)} sentences ({sum(y_true)} positive / {len(y_true) - sum(y_true)} negative)")

    # 2. Paths
    oracle_file = os.path.join(args.results_dir, "baseline_oracle_preds.csv")
    random_file = os.path.join(args.results_dir, "baseline_random_preds.csv")
    app_a_file = os.path.join(args.results_dir, "approachA_mnli_base_predictions.csv")
    app_b_file = os.path.join(args.results_dir, "approachB_selfcheck_predictions.csv")

    runs = []

    # Evaluate Oracle
    if os.path.exists(oracle_file):
        df_oracle = pd.read_csv(oracle_file)
        runs.append(evaluate_single_run(
            "Oracle Baseline",
            y_true,
            df_oracle["hallucination_score"].values,
            threshold=0.50,
            testset_df=testset_df,
            runtime_seconds=0.01
        ))

    # Evaluate Random
    if os.path.exists(random_file):
        df_random = pd.read_csv(random_file)
        runs.append(evaluate_single_run(
            "Random Baseline",
            y_true,
            df_random["hallucination_score"].values,
            threshold=0.50,
            testset_df=testset_df,
            runtime_seconds=0.01
        ))

    # Evaluate Approach A
    if os.path.exists(app_a_file):
        df_a = pd.read_csv(app_a_file)
        scores_a = df_a["hallucination_score"].values
        runtime_a = float(df_a["seconds"].sum()) if "seconds" in df_a.columns else 107.01
        
        # Default threshold (0.50)
        runs.append(evaluate_single_run(
            "Approach A (SummaC NLI - Default)",
            y_true,
            scores_a,
            threshold=0.50,
            testset_df=testset_df,
            runtime_seconds=runtime_a
        ))
        
        # Domain-calibrated threshold (0.85)
        runs.append(evaluate_single_run(
            "Approach A (SummaC NLI - Calibrated)",
            y_true,
            scores_a,
            threshold=0.85,
            testset_df=testset_df,
            runtime_seconds=runtime_a
        ))

    # Evaluate Approach B
    if os.path.exists(app_b_file):
        df_b = pd.read_csv(app_b_file)
        scores_b = df_b["hallucination_score"].values
        runtime_b = float(df_b["seconds"].sum()) if "seconds" in df_b.columns else 0.51
        
        # Default threshold (0.50)
        runs.append(evaluate_single_run(
            "Approach B (SelfCheckGPT)",
            y_true,
            scores_b,
            threshold=0.50,
            testset_df=testset_df,
            runtime_seconds=runtime_b
        ))

    # 3. Compute and Evaluate Hybrid Ensemble
    if os.path.exists(app_a_file) and os.path.exists(app_b_file):
        df_a = pd.read_csv(app_a_file)
        df_b = pd.read_csv(app_b_file)
        
        # Merge on row_id
        merged = testset_df[["row_id"]].merge(df_a[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_a"}), on="row_id")
        merged = merged.merge(df_b[["row_id", "hallucination_score"]].rename(columns={"hallucination_score": "score_b"}), on="row_id")
        
        hybrid_scores = compute_hybrid_scores(
            merged["score_a"].values,
            merged["score_b"].values,
            method="calibrated_max",
            theta_a=0.85
        )
        
        # Save hybrid predictions CSV
        hybrid_csv_path = os.path.join(args.results_dir, "approach_hybrid_predictions.csv")
        df_hybrid = pd.DataFrame({
            "row_id": merged["row_id"],
            "hallucination_score": np.round(hybrid_scores, 4),
            "seconds": 0.0
        })
        df_hybrid.to_csv(hybrid_csv_path, index=False)
        print(f"Saved hybrid ensemble predictions: {hybrid_csv_path}")
        
        runtime_hybrid = (float(df_a["seconds"].sum()) if "seconds" in df_a.columns else 107.01) + (float(df_b["seconds"].sum()) if "seconds" in df_b.columns else 0.51)
        runs.append(evaluate_single_run(
            "Hybrid (NLI + SelfCheck)",
            y_true,
            hybrid_scores,
            threshold=0.50,
            testset_df=testset_df,
            runtime_seconds=runtime_hybrid
        ))

    # 4. Build DataFrames
    df_metrics = build_comparison_dataframe(runs)
    df_errors = build_error_breakdown_dataframe(runs)

    # 5. Save Summary Outputs
    csv_out = os.path.join(args.results_dir, "comparison_summary.csv")
    md_out = os.path.join(args.results_dir, "comparison_summary.md")
    json_out = os.path.join(args.results_dir, "comparison_summary.json")
    
    df_metrics.to_csv(csv_out, index=False)
    print(f"Saved metrics CSV: {csv_out}")
    
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "runs": runs
        }, f, indent=2)
    print(f"Saved metrics JSON: {json_out}")
    
    # Generate Markdown Report
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Comparative Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Benchmark Dataset:** `{args.testset}` (462 sentences)\n\n")
        f.write("## 1. Overall Performance Metrics\n\n")
        f.write(dataframe_to_markdown(df_metrics))
        f.write("\n\n## 2. Per-Error-Type Recall Breakdown\n\n")
        f.write(dataframe_to_markdown(df_errors))
        f.write("\n")
    print(f"Saved Markdown report: {md_out}")

    # 6. Generate Plot
    if args.plot:
        chart_out = os.path.join(args.results_dir, "comparison_breakdown.png")
        plot_comparison_dashboard(df_metrics, df_errors, chart_out)
        print(f"Saved comparison dashboard chart: {chart_out}")

    # 7. Print Terminal Summary
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("\n" + "=" * 70)
    print("  PER-ERROR-TYPE RECALL BREAKDOWN")
    print("=" * 70)
    print(df_errors.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
