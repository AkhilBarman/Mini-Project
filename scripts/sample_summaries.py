"""
Sample Summaries for Approach B (SelfCheckGPT).

Generates N stochastic alternative summaries per court judgment in the evaluation test set.
Saves samples to: data/sampled_summaries/<pair_id>_samples.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hd.sampling import LLMSampler, save_sampled_summaries
from hd.text_utils import split_sentences


def generate_synthetic_legal_sample(judgment_text: str, original_summary: str, seed_idx: int) -> str:
    """
    Generate realistic alternative legal summary sample for reproducible offline development.
    Selects, re-orders, and paraphrases key holding sentences from the factual text.
    """
    sents = split_sentences(original_summary)
    if not sents:
        sents = split_sentences(judgment_text[:1000])

    # Rephrase / select sentences based on seed_idx
    selected = []
    if seed_idx == 0:
        # Front-loaded holding focus
        selected = [s for i, s in enumerate(sents) if i % 2 == 0 or i < 3]
    elif seed_idx == 1:
        # Evidence / reasoning focus
        selected = [s for i, s in enumerate(sents) if i % 2 != 0 or i == 0]
    else:
        # Full procedural focus
        selected = sents[:min(len(sents), 6)]

    return " ".join(selected)


def sample_all_summaries(
    pairs_jsonl: str,
    testset_csv: str,
    output_dir: str,
    backend: str = "synthetic",
    model_name: str = "llama3",
    n_samples: int = 3,
    temperature: float = 0.7
) -> None:
    """Generate or retrieve sampled summaries for all unique testset pairs."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load testset pair IDs
    df_test = pd.read_csv(testset_csv)
    test_pair_ids = set(df_test["pair_id"].unique())
    print(f"Loaded {len(test_pair_ids)} unique judgment pairs from {testset_csv}")

    # 2. Load judgment pairs
    pairs_data: Dict[str, Dict[str, str]] = {}
    with open(pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] in test_pair_ids:
                pairs_data[item["id"]] = {
                    "judgment_text": item["judgment_text"],
                    "summary_text": item["summary_text"]
                }

    print(f"Matched {len(pairs_data)} judgments for sampling. Backend: {backend.upper()}")

    # 3. Initialize sampler if not synthetic
    sampler = None
    if backend != "synthetic":
        sampler = LLMSampler(
            backend=backend,
            model_name=model_name,
            temperature=temperature
        )

    # 4. Generate samples for each pair
    for pair_id in tqdm(sorted(test_pair_ids), desc="Generating samples"):
        target_file = os.path.join(output_dir, f"{pair_id}_samples.json")
        if os.path.exists(target_file):
            continue

        pair_info = pairs_data.get(pair_id, {})
        j_text = pair_info.get("judgment_text", "")
        s_text = pair_info.get("summary_text", "")

        if backend == "synthetic":
            samples = [
                generate_synthetic_legal_sample(j_text, s_text, idx)
                for idx in range(n_samples)
            ]
        else:
            samples = sampler.sample_summaries(j_text, n_samples=n_samples)

        save_sampled_summaries(pair_id, samples, output_dir=output_dir)

    print(f"\n[OK] Sampled summaries saved to {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Generate stochastic sampled summaries for Approach B.")
    parser.add_argument(
        "--pairs-jsonl",
        default=os.path.join(project_root, "data", "processed", "sample_pairs.jsonl"),
        help="Path to sample_pairs.jsonl"
    )
    parser.add_argument(
        "--testset",
        default=os.path.join(project_root, "data", "testset", "testset.csv"),
        help="Path to testset.csv"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "data", "sampled_summaries"),
        help="Directory to save sampled summaries"
    )
    parser.add_argument(
        "--backend",
        choices=["synthetic", "ollama", "transformers"],
        default="synthetic",
        help="Sampling backend (default: synthetic)"
    )
    parser.add_argument(
        "--model-name",
        default="llama3",
        help="LLM model name (default: llama3)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of stochastic samples per judgment (default: 3)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )

    args = parser.parse_args()
    sample_all_summaries(
        pairs_jsonl=os.path.abspath(args.pairs_jsonl),
        testset_csv=os.path.abspath(args.testset),
        output_dir=os.path.abspath(args.output_dir),
        backend=args.backend,
        model_name=args.model_name,
        n_samples=args.n_samples,
        temperature=args.temp
    )


if __name__ == "__main__":
    main()
