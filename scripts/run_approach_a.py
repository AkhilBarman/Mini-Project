"""
Run Approach A (SummaC NLI) on the evaluation test set.

Generates sentence-level predictions:
  results/approachA_<model>_predictions.csv
Columns:
  row_id, hallucination_score, seconds
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.approach_a import NLIChecker
from hd.evaluation import Timer


def get_hardware_info() -> Dict[str, str]:
    """Retrieve basic hardware specifications for execution metadata."""
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version()
    }
    return info


def run_approach_a(
    testset_path: str,
    pairs_jsonl_path: str,
    output_csv_path: str,
    model_name: str = "mnli-base",
    device: str = "cpu",
    max_doc_sents: int = 100,
    max_pairs: int = -1
) -> str:
    """Run NLIChecker across testset summaries and save predictions CSV."""
    if not os.path.exists(testset_path):
        print(f"Error: Test set CSV not found at {testset_path}")
        sys.exit(1)

    if not os.path.exists(pairs_jsonl_path):
        print(f"Error: Processed pairs JSONL not found at {pairs_jsonl_path}")
        sys.exit(1)

    # 1. Load judgment texts
    judgments: Dict[str, str] = {}
    with open(pairs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            judgments[item["id"]] = item["judgment_text"]

    # 2. Load test set
    df_test = pd.read_csv(testset_path)
    total_rows = len(df_test)

    # Filter pairs if max_pairs is specified
    if max_pairs > 0:
        unique_pairs = df_test["pair_id"].unique()[:max_pairs]
        df_test = df_test[df_test["pair_id"].isin(unique_pairs)].copy()
        print(f"Filtered to first {max_pairs} pairs ({len(df_test)} / {total_rows} rows).")

    # 3. Group by (pair_id, version) to score summary documents coherently
    grouped = df_test.groupby(["pair_id", "version"], sort=False)
    n_groups = len(grouped)

    print("=" * 70)
    print(f"APPROACH A (SummaC NLI) INFERENCE RUN")
    print(f"Model Backend:       {model_name}")
    print(f"Device:              {device}")
    print(f"Testset:             {testset_path} ({len(df_test)} sentences across {n_groups} summaries)")
    print(f"Output File:         {output_csv_path}")
    print("=" * 70)

    # 4. Initialize checker
    checker = NLIChecker(
        model_name=model_name,
        device=device,
        max_doc_sents=max_doc_sents
    )

    pred_records: List[Dict[str, Any]] = []
    summary_timings: List[float] = []

    with Timer("total_approach_a_run") as t_total:
        pbar = tqdm(grouped, desc="Scoring summaries", unit="summary")
        for (pid, ver), group_df in pbar:
            j_text = judgments[pid]
            sentences = group_df["sentence_text"].tolist()
            row_ids = group_df["row_id"].tolist()

            t0 = time.time()
            h_scores = checker.score_sentences(j_text, sentences)
            t_elapsed = time.time() - t0
            summary_timings.append(t_elapsed)

            sec_per_sent = round(t_elapsed / max(len(sentences), 1), 6)
            for r_id, score in zip(row_ids, h_scores):
                pred_records.append({
                    "row_id": r_id,
                    "hallucination_score": round(score, 6),
                    "seconds": sec_per_sent
                })

            pbar.set_postfix({"last_sum_sec": f"{t_elapsed:.1f}s", "n_sents": len(sentences)})

    # 5. Write predictions CSV
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
    print(f"Average Time per Sentence:   {avg_sec_per_sentence:.2f} seconds")
    print("=" * 70 + "\n")

    return output_csv_path


def main():
    parser = argparse.ArgumentParser(description="Run Approach A (SummaC NLI) on test set.")
    parser.add_argument(
        "--testset",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to testset.csv"
    )
    parser.add_argument(
        "--pairs-jsonl",
        default=os.path.join(project_root, "data", "processed", "sample_pairs.jsonl"),
        help="Path to sample_pairs.jsonl"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(project_root, "results", "approachA_mnli_base_predictions.csv"),
        help="Output predictions CSV path"
    )
    parser.add_argument(
        "--model-name",
        default="mnli-base",
        help="SummaC model name (default: mnli-base)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--max-doc-sents",
        type=int,
        default=100,
        help="Max document sentences to score against (default: 100)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=-1,
        help="Limit to first N pairs (-1 for all)"
    )
    args = parser.parse_args()

    run_approach_a(
        testset_path=os.path.abspath(args.testset),
        pairs_jsonl_path=os.path.abspath(args.pairs_jsonl),
        output_csv_path=os.path.abspath(args.out),
        model_name=args.model_name,
        device=args.device,
        max_doc_sents=args.max_doc_sents,
        max_pairs=args.max_pairs
    )


if __name__ == "__main__":
    main()
