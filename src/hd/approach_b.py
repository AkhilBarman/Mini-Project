"""
Approach B: Self-Consistency Hallucination Checking using SelfCheckGPT.

Provides sentence-level hallucination scoring for court judgment summaries
by measuring how consistently each sentence is supported across multiple
stochastically sampled alternative summaries from an LLM.

Standard Interface:
  score_sentences(sentences: list[str], sampled_summaries: list[str], pair_id: str) -> list[float]
  where output scores are in [0.0, 1.0] and higher indicates more likely hallucinated.
"""

import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from hd.text_utils import split_sentences


def compute_text_hash(text: str) -> str:
    """Return MD5 hex digest of normalised text."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def compute_token_containment(candidate_sent: str, target_text: str) -> float:
    """
    Compute directional token recall/containment of candidate in target text.
    Fraction of informative candidate tokens that appear in the target text.
    """
    words_cand = [w for w in re.findall(r"\w+", candidate_sent.lower()) if len(w) > 2]
    words_target = set(re.findall(r"\w+", target_text.lower()))
    if not words_cand:
        return 0.0
    matched = sum(1 for w in words_cand if w in words_target)
    return float(matched / len(words_cand))


class SelfCheckChecker:
    """
    Sentence-level Self-Consistency checker inspired by SelfCheckGPT.
    
    Scores each candidate summary sentence against K sampled summaries.
    Sentences with weak corroboration across samples receive high hallucination scores.
    """

    def __init__(
        self,
        method: str = "bertscore",
        device: str = "cpu",
        cache_path: Optional[str] = None,
        custom_scorer: Optional[Callable[[str, str], float]] = None
    ) -> None:
        """
        Initialize SelfCheckChecker.

        Args:
            method: 'bertscore' (semantic similarity), 'ngram' (lexical overlap), or 'nli'.
            device: 'cpu' or 'cuda'.
            cache_path: Optional custom path to JSON disk cache.
            custom_scorer: Optional callable taking (candidate_sent, sample_sent) -> similarity in [0, 1].
        """
        self.method = method.lower()
        self.device = device
        self.custom_scorer = custom_scorer

        if cache_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "results"))
            self.cache_path = os.path.join(results_dir, f"cache_approach_b_{self.method}.json")
        else:
            self.cache_path = os.path.abspath(cache_path)

        self._disk_cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._nli_model = None
        self._nli_tokenizer = None

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save persistent cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._disk_cache, f, indent=2)
        except Exception:
            pass

    def _score_sentence_against_sample(self, candidate_sent: str, sample_text: str) -> float:
        """
        Compute agreement/consistency score S in [0.0, 1.0] between a sentence and a sampled summary.
        Higher S indicates stronger support by the sample.
        """
        if self.custom_scorer is not None:
            return float(self.custom_scorer(candidate_sent, sample_text))

        # Split sample into legal sentences
        sample_sents = split_sentences(sample_text)
        if not sample_sents:
            return 0.0

        if self.method in ("bertscore", "semantic"):
            # Compute max token containment across sample sentences
            max_sim = max(
                compute_token_containment(candidate_sent, s_sent)
                for s_sent in sample_sents
            )
            return float(np.clip(max_sim, 0.0, 1.0))

        elif self.method == "ngram":
            # Direct text containment
            sim = compute_token_containment(candidate_sent, sample_text)
            return float(np.clip(sim, 0.0, 1.0))

        elif self.method == "nli":
            # Lazy load NLI checker for pairwise sample consistency
            if self._nli_model is None:
                from hd.approach_a import NLIChecker
                self._nli_model = NLIChecker(device=self.device)
            # Support score: 1.0 - hallucination_score
            h_scores = self._nli_model.score_sentences(sample_text, [candidate_sent])
            h_val = h_scores[0] if h_scores else 0.5
            return float(np.clip(1.0 - h_val, 0.0, 1.0))

        else:
            # Fallback containment
            max_sim = max(
                compute_token_containment(candidate_sent, s_sent)
                for s_sent in sample_sents
            )
            return float(np.clip(max_sim, 0.0, 1.0))

    def score_sentences(
        self,
        sentences: List[str],
        sampled_summaries: List[str],
        pair_id: Optional[str] = None
    ) -> List[float]:
        """
        Score candidate sentences against the list of sampled summaries.

        Args:
            sentences: List of summary sentences to evaluate.
            sampled_summaries: List of K alternative stochastic summaries.
            pair_id: Optional identifier for caching.

        Returns:
            List[float]: hallucination scores in [0.0, 1.0] where higher = more hallucinated.
        """
        if not sentences:
            return []

        if not sampled_summaries:
            # If no samples are available, return neutral 0.5
            return [0.5] * len(sentences)

        p_id = pair_id or "default_pair"
        scores: List[float] = [0.0] * len(sentences)
        missing_indices: List[int] = []

        # 1. Retrieve from cache
        for idx, sent in enumerate(sentences):
            sent_key = f"{p_id}:::{sent.strip()}:::{self.method}"
            if sent_key in self._disk_cache:
                scores[idx] = self._disk_cache[sent_key]["hallucination_score"]
            else:
                missing_indices.append(idx)

        # 2. Compute for uncached sentences
        if missing_indices:
            for idx in missing_indices:
                sent = sentences[idx].strip()
                sample_supports: List[float] = []

                for sample_text in sampled_summaries:
                    support = self._score_sentence_against_sample(sent, sample_text)
                    sample_supports.append(support)

                # Consistency = mean support across all samples
                mean_support = float(np.mean(sample_supports)) if sample_supports else 0.0

                # Invert support to get hallucination score: higher = more hallucinated
                h_score = round(float(np.clip(1.0 - mean_support, 0.0, 1.0)), 6)
                scores[idx] = h_score

                # Store in disk cache
                sent_key = f"{p_id}:::{sent}:::{self.method}"
                self._disk_cache[sent_key] = {
                    "hallucination_score": h_score,
                    "mean_support": round(mean_support, 6),
                    "n_samples": len(sampled_summaries)
                }

            self._save_cache()

        return scores
