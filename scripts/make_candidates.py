"""
Generate candidate sentence mutations for human review.

Iterates over data/processed/sample_pairs.jsonl, selects 1-2 candidate sentences
per summary across the 5 error types, and outputs:
  - data/testset/candidates.csv (with accept, fixed_sentence, and notes blank for review)
  - data/testset/candidates_UNREVIEWED.csv (development copy with accept='Y')
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# Ensure src/ is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.mutations import (
    ERROR_TYPES,
    get_applicable_mutations,
    mutate_sentence
)


def generate_candidates(
    input_jsonl: str,
    output_dir: str,
    seed: int = 42,
    target_edits_per_doc: int = 2
) -> List[Dict[str, str]]:
    """Generate balanced candidate mutations across summary sentences."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    if not os.path.exists(input_jsonl):
        print(f"Error: Processed pairs file not found at {input_jsonl}")
        print("Please run: python scripts/prepare_data.py first.")
        sys.exit(1)

    pairs = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} document pairs from {input_jsonl}")

    candidates: List[Dict[str, str]] = []
    type_counts = {t: 0 for t in ERROR_TYPES}
    cand_counter = 1

    for pair in pairs:
        pair_id = pair["id"]
        s_sents = pair["summary_sentences"]
        if not s_sents:
            print(f"Skipping {pair_id}: No summary sentences found.")
            continue

        # Find all valid (sentence_idx, error_type, edited_text, desc) possibilities
        valid_candidates_for_doc: List[Tuple[int, str, str, str]] = []

        for idx, sent in enumerate(s_sents):
            # Exclude very short sentences or title headers (e.g. "Title: ...")
            if len(sent.split()) < 4 or sent.strip().lower().startswith("title:"):
                continue

            app = get_applicable_mutations(sent, seed=seed + idx)
            for etype, (edited, desc) in app.items():
                if edited.strip() != sent.strip():
                    valid_candidates_for_doc.append((idx, etype, edited, desc))

        if not valid_candidates_for_doc:
            print(f"Warning: No valid mutations applicable for {pair_id}.")
            continue

        # Shuffle candidates deterministically
        rng.shuffle(valid_candidates_for_doc)

        # Select up to target_edits_per_doc, preferring less-represented error types
        # and distinct sentence indices
        selected_for_doc: List[Tuple[int, str, str, str]] = []
        used_indices = set()

        # Sort candidates for this doc by current global frequency of error_type (least used first)
        sorted_candidates = sorted(
            valid_candidates_for_doc,
            key=lambda c: (type_counts[c[1]], c[0])
        )

        for s_idx, etype, edited, desc in sorted_candidates:
            if len(selected_for_doc) >= target_edits_per_doc:
                break
            if s_idx in used_indices:
                continue

            selected_for_doc.append((s_idx, etype, edited, desc))
            used_indices.add(s_idx)
            type_counts[etype] += 1

        # Format rows
        for s_idx, etype, edited, desc in selected_for_doc:
            cand_id = f"cand_{cand_counter:03d}"
            cand_counter += 1
            candidates.append({
                "candidate_id": cand_id,
                "pair_id": pair_id,
                "sentence_idx": str(s_idx),
                "original_sentence": s_sents[s_idx],
                "edited_sentence": edited,
                "error_type": etype,
                "accept": "",
                "fixed_sentence": "",
                "notes": desc
            })

    # Output paths
    candidates_csv = os.path.join(output_dir, "candidates.csv")
    unreviewed_csv = os.path.join(output_dir, "candidates_UNREVIEWED.csv")

    fieldnames = [
        "candidate_id", "pair_id", "sentence_idx", "original_sentence",
        "edited_sentence", "error_type", "accept", "fixed_sentence", "notes"
    ]

    # Write blank review copy
    with open(candidates_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    # Write unreviewed development copy (with accept='Y')
    with open(unreviewed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            c_dev = dict(c)
            c_dev["accept"] = "Y"
            writer.writerow(c_dev)

    print(f"\n[OK] Generated {len(candidates)} candidate mutations.")
    print(f"Review file saved to: {candidates_csv}")
    print(f"Development copy (UNREVIEWED) saved to: {unreviewed_csv}\n")

    print("Breakdown by Error Type:")
    for etype in ERROR_TYPES:
        print(f"  - {etype:<15}: {type_counts[etype]} candidates")

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Generate candidate sentence mutations for review.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--input",
        default=os.path.join(project_root, "data", "processed", "sample_pairs.jsonl"),
        help="Path to sample_pairs.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "data", "testset"),
        help="Path to output directory for testset files"
    )
    parser.add_argument(
        "--edits-per-doc",
        type=int,
        default=2,
        help="Number of mutated sentences per summary (default: 2)"
    )
    args = parser.parse_args()

    generate_candidates(
        input_jsonl=os.path.abspath(args.input),
        output_dir=os.path.abspath(args.output_dir),
        seed=args.seed,
        target_edits_per_doc=args.edits_per_doc
    )


if __name__ == "__main__":
    main()
