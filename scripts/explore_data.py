"""
Command-line exploratory data analysis for processed MILDSum pairs.

Displays:
- Statistical summary of length distributions (chars, words, sentences).
- Side-by-side view of 3 representative judgment-summary pairs.
- Discussion of downstream challenges (SummaC token limits, chunking, etc.).
"""

import json
import os
import sys
import pandas as pd

def explore():
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    stats_path = os.path.join(processed_dir, "stats.csv")
    jsonl_path = os.path.join(processed_dir, "sample_pairs.jsonl")

    if not os.path.exists(stats_path) or not os.path.exists(jsonl_path):
        print("Processed data not found. Please run scripts/prepare_data.py first.")
        sys.exit(1)

    df_stats = pd.read_csv(stats_path)
    print("=" * 80)
    print(f"EXPLORATORY DATA ANALYSIS: {len(df_stats)} MILDSUM PAIRS")
    print("=" * 80)

    # Metrics
    metrics = [
        "judgment_chars", "judgment_words", "judgment_sentences",
        "summary_chars", "summary_words", "summary_sentences",
        "compression_ratio_words", "compression_ratio_sents"
    ]
    summary = df_stats[metrics].describe().T[["mean", "std", "min", "50%", "max"]]
    summary.rename(columns={"50%": "median"}, inplace=True)
    print("\nLENGTH AND COMPRESSION DISTRIBUTIONS:")
    print(summary.to_string())

    # Read JSONL
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE INSPECTION OF 3 EXAMPLE PAIRS")
    print("=" * 80)

    for item in pairs[:3]:
        pid = item["id"]
        j_text = item["judgment_text"]
        s_text = item["summary_text"]
        j_sents = item["judgment_sentences"]
        s_sents = item["summary_sentences"]

        print(f"\n--- PAIR ID: {pid} ---")
        print(f"Judgment Length: {len(j_text.split()):,} words | {len(j_sents)} sentences")
        print(f"Summary Length:  {len(s_text.split()):,} words | {len(s_sents)} sentences")
        print("\nJUDGMENT LEAD SENTENCES:")
        for idx, s in enumerate(j_sents[:2], 1):
            print(f"  [{idx}] {s.strip()}")
        print("\nSUMMARY LEAD SENTENCES:")
        for idx, s in enumerate(s_sents[:3], 1):
            print(f"  [{idx}] {s.strip()}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("DOWNSTREAM CHALLENGES & CONSIDERATIONS FOR PHASES 2 - 5")
    print("=" * 80)
    print("1. Maximum Context Length in SummaC / NLI:")
    print(f"   - Max judgment length is {df_stats['judgment_words'].max():,} words (~8,000 tokens).")
    print("   - NLI models (RoBERTa / DeBERTa) have a 512-token limit.")
    print("   - In Phase 3 (Approach A), we must chunk judgments and match sentences.")
    print("2. Legal Formatting & Currency:")
    print("   - Text features Indian currency (Rs. 24,02,380) and section citations.")
    print("   - Our custom NLTK Legal Punkt segmenter preserves these boundaries accurately.")
    print("3. Factual Claims in Summaries:")
    print("   - Summaries heavily cite figures, hospital names, dates, and order directives.")
    print("   - These specific entities are the highest-risk hallucination targets.")
    print("=" * 80)


if __name__ == "__main__":
    explore()
