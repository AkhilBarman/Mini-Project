"""
Build the final ground-truth test set for sentence-level hallucination evaluation.

Merges human-reviewed candidate mutations with original summary sentences from
data/processed/sample_pairs.jsonl to produce:
  data/testset/testset.csv

For every document pair, includes:
  1. The ORIGINAL summary: all sentences labeled 0 (supported).
  2. The EDITED summary: mutated sentences labeled 1 (hallucinated), unedited labeled 0.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def build_testset(
    review_csv_path: str,
    pairs_jsonl_path: str,
    output_csv_path: str
) -> List[Dict[str, str]]:
    """Construct sentence-level testset.csv from reviewed candidates."""
    if not os.path.exists(review_csv_path):
        print(f"Error: Review CSV not found at {review_csv_path}")
        sys.exit(1)

    if not os.path.exists(pairs_jsonl_path):
        print(f"Error: Processed pairs JSONL not found at {pairs_jsonl_path}")
        sys.exit(1)

    # 1. Load accepted mutations from review CSV
    accepted_mutations: Dict[str, Dict[int, Dict[str, str]]] = {}
    total_candidates = 0
    accepted_count = 0

    with open(review_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_candidates += 1
            accept_val = row.get("accept", "").strip().upper()
            if accept_val == "Y":
                accepted_count += 1
                pid = row["pair_id"]
                s_idx = int(row["sentence_idx"])
                if pid not in accepted_mutations:
                    accepted_mutations[pid] = {}
                
                # Use fixed_sentence if provided by reviewer, else edited_sentence
                final_text = row.get("fixed_sentence", "").strip()
                if not final_text:
                    final_text = row["edited_sentence"].strip()

                accepted_mutations[pid][s_idx] = {
                    "candidate_id": row["candidate_id"],
                    "edited_text": final_text,
                    "original_text": row["original_sentence"].strip(),
                    "error_type": row["error_type"].strip(),
                    "notes": row.get("notes", "").strip()
                }

    print(f"Loaded review file: {review_csv_path}")
    print(f"Total candidates reviewed: {total_candidates} | Accepted: {accepted_count}")

    if accepted_count == 0:
        print("Warning: 0 candidates were accepted (accept = 'Y').")
        print("If you haven't reviewed candidates.csv yet, use: --from-review data/testset/candidates_UNREVIEWED.csv")

    # 2. Load pairs
    pairs = []
    with open(pairs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    # 3. Build testset rows
    testset_rows: List[Dict[str, str]] = []
    type_counts: Dict[str, int] = {}
    pos_count = 0
    neg_count = 0

    for pair in pairs:
        pair_id = pair["id"]
        s_sents = pair["summary_sentences"]
        pair_mutations = accepted_mutations.get(pair_id, {})

        # --- A. Original Summary Version (all label 0) ---
        for idx, sent in enumerate(s_sents):
            row_id = f"{pair_id}_orig_s{idx:02d}"
            testset_rows.append({
                "row_id": row_id,
                "pair_id": pair_id,
                "version": "original",
                "sentence_idx": str(idx),
                "sentence_text": sent.strip(),
                "label": "0",
                "error_type": "",
                "original_sentence": ""
            })
            neg_count += 1

        # --- B. Edited Summary Version (mutated = 1, unchanged = 0) ---
        for idx, sent in enumerate(s_sents):
            row_id = f"{pair_id}_edited_s{idx:02d}"
            if idx in pair_mutations:
                mut = pair_mutations[idx]
                testset_rows.append({
                    "row_id": row_id,
                    "pair_id": pair_id,
                    "version": "edited",
                    "sentence_idx": str(idx),
                    "sentence_text": mut["edited_text"],
                    "label": "1",
                    "error_type": mut["error_type"],
                    "original_sentence": mut["original_text"]
                })
                pos_count += 1
                type_counts[mut["error_type"]] = type_counts.get(mut["error_type"], 0) + 1
            else:
                testset_rows.append({
                    "row_id": row_id,
                    "pair_id": pair_id,
                    "version": "edited",
                    "sentence_idx": str(idx),
                    "sentence_text": sent.strip(),
                    "label": "0",
                    "error_type": "",
                    "original_sentence": ""
                })
                neg_count += 1

    # 4. Save testset.csv
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        "row_id", "pair_id", "version", "sentence_idx",
        "sentence_text", "label", "error_type", "original_sentence"
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(testset_rows)

    print("\n" + "=" * 60)
    print("TESTSET CONSTRUCTION SUMMARY")
    print("=" * 60)
    print(f"Output File:                 {os.path.abspath(output_csv_path)}")
    print(f"Total Document Pairs:        {len(pairs)}")
    print(f"Total Sentence Rows:         {len(testset_rows)}")
    print(f"Negative Rows (Supported/0): {neg_count}")
    print(f"Positive Rows (Halluc./1):   {pos_count}")
    print(f"Imbalance Ratio (Pos/Total): {pos_count / max(len(testset_rows), 1) * 100:.1f}%\n")

    print("Positive Rows by Error Type:")
    for etype, count in sorted(type_counts.items()):
        print(f"  - {etype:<15}: {count} rows")
    print("=" * 60 + "\n")

    return testset_rows


def main():
    parser = argparse.ArgumentParser(description="Build sentence-level evaluation testset.")
    parser.add_argument(
        "--from-review",
        default=os.path.join(project_root, "data", "testset", "candidates_UNREVIEWED.csv"),
        help="Path to reviewed candidates CSV (default: data/testset/candidates_UNREVIEWED.csv)"
    )
    parser.add_argument(
        "--input-jsonl",
        default=os.path.join(project_root, "data", "processed", "sample_pairs.jsonl"),
        help="Path to sample_pairs.jsonl"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to output testset.csv"
    )
    args = parser.parse_args()

    build_testset(
        review_csv_path=os.path.abspath(args.from_review),
        pairs_jsonl_path=os.path.abspath(args.input_jsonl),
        output_csv_path=os.path.abspath(args.output)
    )


if __name__ == "__main__":
    main()
