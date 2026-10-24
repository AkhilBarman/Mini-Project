"""
Prepare a clean, sampled dataset of Indian legal judgments and summaries.

Loads raw pairs, normalises whitespace, segments into sentences using legal
sentence boundary rules, samples N pairs reproducibly, and outputs:
  - data/processed/sample_pairs.jsonl
  - data/processed/stats.csv
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
from tqdm import tqdm
from hd.data_loader import load_mildsum_pairs
from hd.text_utils import clean_text, split_sentences


def compute_counts(text: str, sentences: List[str]) -> Dict[str, int]:
    """Compute character, word, and sentence counts."""
    chars = len(text)
    words = len(text.split())
    sents = len(sentences)
    return {
        "chars": chars,
        "words": words,
        "sentences": sents
    }


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    n_samples: int = 30,
    seed: int = 42,
    method: str = "nltk_legal"
) -> None:
    """Load, process, sample, and save judgment-summary pairs.

    Args:
        data_dir: Directory containing raw sample pairs.
        output_dir: Directory to save processed JSONL and CSV files.
        n_samples: Target sample size (reproducible with seed).
        seed: Random seed for reproducibility.
        method: Sentence splitting method ('nltk_legal' or 'pysbd').
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw pairs from: {os.path.abspath(data_dir)}")
    pairs = load_mildsum_pairs(data_dir=data_dir, verbose=True)

    if not pairs:
        print(f"\nNo valid pairs found in {data_dir}.")
        print("Attempting to auto-download available MILDSum samples...")
        try:
            from download_mildsum import download_mildsum_samples
            download_mildsum_samples(data_dir)
            pairs = load_mildsum_pairs(data_dir=data_dir, verbose=True)
        except Exception as e:
            print(f"Could not auto-download: {e}")
            print(f"Please run: python scripts/download_mildsum.py")
            return

    if not pairs:
        print("Error: No data pairs available to prepare.")
        return

    print(f"\nProcessing and segmenting {len(pairs)} document pairs...")
    processed_records: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    for item in tqdm(pairs, desc="Segmenting"):
        pair_id = item["id"]
        j_cleaned = clean_text(item["judgment_text"])
        s_cleaned = clean_text(item["summary_text"])

        j_sents = split_sentences(j_cleaned, method=method)
        s_sents = split_sentences(s_cleaned, method=method)

        # Skip completely empty texts
        if not j_sents or not s_sents:
            continue

        j_counts = compute_counts(j_cleaned, j_sents)
        s_counts = compute_counts(s_cleaned, s_sents)

        rec = {
            "id": pair_id,
            "judgment_text": j_cleaned,
            "summary_text": s_cleaned,
            "judgment_sentences": j_sents,
            "summary_sentences": s_sents
        }
        processed_records.append(rec)

        row = {
            "id": pair_id,
            "judgment_chars": j_counts["chars"],
            "judgment_words": j_counts["words"],
            "judgment_sentences": j_counts["sentences"],
            "summary_chars": s_counts["chars"],
            "summary_words": s_counts["words"],
            "summary_sentences": s_counts["sentences"],
            "compression_ratio_words": round(s_counts["words"] / max(j_counts["words"], 1), 4),
            "compression_ratio_sents": round(s_counts["sentences"] / max(j_counts["sentences"], 1), 4)
        }
        stats_rows.append(row)

    # Sampling strategy:
    # 1. Prefer summaries with at least 3 sentences
    # 2. Reproducibly sample N items using random seed
    random.seed(seed)
    high_quality = [r for r in processed_records if len(r["summary_sentences"]) >= 3]
    low_quality = [r for r in processed_records if len(r["summary_sentences"]) < 3]

    combined_pool = high_quality + low_quality

    if len(combined_pool) < n_samples:
        print(f"\nNotice: Requested n={n_samples}, but only {len(combined_pool)} valid pairs exist in raw data.")
        print(f"Sampling all {len(combined_pool)} available pairs.")
        sampled_records = combined_pool
    else:
        # Shuffle deterministically and pick top N
        random.shuffle(combined_pool)
        sampled_records = combined_pool[:n_samples]

    # Sort sampled records by id for stable output ordering
    def _id_sort_key(r):
        parts = r["id"].split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return (parts[0], int(parts[-1]))
        return (r["id"], 0)

    sampled_records = sorted(sampled_records, key=_id_sort_key)
    sampled_ids = {r["id"] for r in sampled_records}
    sampled_stats = [row for row in stats_rows if row["id"] in sampled_ids]
    sampled_stats = sorted(sampled_stats, key=lambda r: _id_sort_key(r))

    # Save sample_pairs.jsonl
    jsonl_path = os.path.join(output_dir, "sample_pairs.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in sampled_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved {len(sampled_records)} pairs to {jsonl_path}")

    # Save stats.csv
    csv_path = os.path.join(output_dir, "stats.csv")
    df_stats = pd.DataFrame(sampled_stats)
    df_stats.to_csv(csv_path, index=False)
    print(f"[OK] Saved statistics table to {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 65)
    print("PHASE 1 SAMPLE SUMMARY STATISTICS")
    print("=" * 65)
    print(f"Total Sampled Pairs:            {len(df_stats)}")
    print(f"Avg Judgment Characters:        {df_stats['judgment_chars'].mean():.1f}")
    print(f"Avg Judgment Words:             {df_stats['judgment_words'].mean():.1f}")
    print(f"Avg Judgment Sentences:         {df_stats['judgment_sentences'].mean():.1f}")
    print(f"Avg Summary Characters:         {df_stats['summary_chars'].mean():.1f}")
    print(f"Avg Summary Words:              {df_stats['summary_words'].mean():.1f}")
    print(f"Avg Summary Sentences:          {df_stats['summary_sentences'].mean():.1f}")
    print(f"Avg Word Compression Ratio:     {df_stats['compression_ratio_words'].mean() * 100:.1f}%")
    print(f"Avg Sentence Compression Ratio: {df_stats['compression_ratio_sents'].mean() * 100:.1f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Prepare and sample MILDSum pairs.")
    parser.add_argument("--n", type=int, default=30, help="Target sample size (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
        help="Path to processed output directory"
    )
    parser.add_argument(
        "--method",
        default="nltk_legal",
        choices=["nltk_legal", "pysbd"],
        help="Sentence splitting method (default: nltk_legal)"
    )
    args = parser.parse_args()

    prepare_dataset(
        data_dir=os.path.abspath(args.data_dir),
        output_dir=os.path.abspath(args.output_dir),
        n_samples=args.n,
        seed=args.seed,
        method=args.method
    )


if __name__ == "__main__":
    main()
