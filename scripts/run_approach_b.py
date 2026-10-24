"""
Run Approach B (SelfCheckGPT Consistency Checking) on the evaluation test set.

Generates sentence-level predictions:
  results/approachB_<method>_predictions.csv
Columns:
  row_id, hallucination_score, seconds
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.approach_b import SelfCheckChecker
from hd.sampling import load_sampled_summaries
from hd.evaluation import Timer


def run_approach_b(
    testset_path: str,
    samples_dir: str,
    output_csv_path: str,
    method: str = "bertscore",
    device: str = "cpu"
) -> str:
    """Run SelfCheckChecker across testset summaries and save predictions CSV."""
    if not os.path.exists(testset_path):
        print(f"Error: Test set CSV not found at {testset_path}")
        sys.exit(1)

    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found at {samples_dir}")
        sys.exit(1)

    df_test = pd.read_csv(testset_path)
    grouped = df_test.groupby(["pair_id", "version"], sort=False)
    n_groups = len(grouped)

    print("=" * 70)
    print("APPROACH B (SelfCheckGPT Consistency) INFERENCE RUN")
    print(f"Method:              {method}")
    print(f"Device:              {device}")
    print(f"Testset:             {testset_path} ({len(df_test)} sentences across {n_groups} summaries)")
    print(f"Samples Directory:   {samples_dir}")
    print(f"Output File:         {output_csv_path}")
    print("=" * 70)

    checker = SelfCheckChecker(method=method, device=device)

    pred_records: List[Dict[str, Any]] = []
    summary_timings: List[float] = []

    with Timer("total_approach_b_run") as t_total:
        pbar = tqdm(grouped, desc="Scoring summaries", unit="summary")
        for (pid, ver), group_df in pbar:
            sentences = group_df["sentence_text"].tolist()
            row_ids = group_df["row_id"].tolist()

            # Load alternative sampled summaries for this judgment
            sampled_summaries = load_sampled_summaries(pid, input_dir=samples_dir)

            t0 = time.time()
            h_scores = checker.score_sentences(
                sentences=sentences,
                sampled_summaries=sampled_summaries,
                pair_id=pid
            )
            t_elapsed = time.time() - t0
            summary_timings.append(t_elapsed)

            sec_per_sent = round(t_elapsed / max(len(sentences), 1), 6)
            for r_id, score in zip(row_ids, h_scores):
                pred_records.append({
                    "row_id": r_id,
                    "hallucination_score": round(score, 6),
                    "seconds": sec_per_sent
                })

            pbar.set_postfix({"last_sum_sec": f"{t_elapsed:.2f}s", "n_sents": len(sentences)})

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(output_csv_path, index=False)

    avg_sec_per_summary = sum(summary_timings) / max(len(summary_timings), 1)
    avg_sec_per_sentence = t_total.elapsed_seconds / max(len(df_preds), 1)

    print("\n" + "=" * 70)
    print("INFERENCE RUN COMPLETE")
    print("=" * 70)
    print(f"Predictions File:            {os.path.abspath(output_csv_path)}")
    print(f"Total Sentences Scored:      {len(df_preds)}")
    print(f"Total Run Time:              {t_total.elapsed_seconds:.2f} seconds ({t_total.elapsed_seconds/60:.2f} minutes)")
    print(f"Average Time per Summary:    {avg_sec_per_summary:.2f} seconds")
    print(f"Average Time per Sentence:   {avg_sec_per_sentence:.4f} seconds")
    print("=" * 70 + "\n")

    return output_csv_path


def main():
    parser = argparse.ArgumentParser(description="Run Approach B (SelfCheckGPT) on test set.")
    parser.add_argument(
        "--testset",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to testset.csv"
    )
    parser.add_argument(
        "--samples-dir",
        default=os.path.join(project_root, "data", "sampled_summaries"),
        help="Path to sampled summaries directory"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(project_root, "results", "approachB_selfcheck_predictions.csv"),
        help="Output predictions CSV path"
    )
    parser.add_argument(
        "--method",
        default="bertscore",
        choices=["bertscore", "ngram", "nli"],
        help="SelfCheck consistency method (default: bertscore)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu or cuda (default: cpu)"
    )

    args = parser.parse_args()
    run_approach_b(
        testset_path=os.path.abspath(args.testset),
        samples_dir=os.path.abspath(args.samples_dir),
        output_csv_path=os.path.abspath(args.out),
        method=args.method,
        device=args.device
    )


if __name__ == "__main__":
    main()
