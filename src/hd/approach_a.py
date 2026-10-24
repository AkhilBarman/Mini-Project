"""
Approach A: NLI-based Hallucination Detection using SummaC.

Provides sentence-level hallucination scoring for court judgment summaries:
- Computes entailment and contradiction probabilities against source judgment chunks.
- Maps SummaC support scores S in [-1.0, 1.0] to hallucination scores in [0.0, 1.0].
- Implements disk caching keyed by (judgment_hash, sentence_text, model_name).
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from summac.model_summac import SummaCImager
except ImportError:
    SummaCImager = None


def compute_text_hash(text: str) -> str:
    """Return MD5 hex digest of normalised text."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


class NLIChecker:
    """
    Sentence-level NLI hallucination checker using SummaC.
    
    Standard Interface:
      score_sentences(judgment_text: str, sentences: list[str]) -> list[float]
      where output is in [0.0, 1.0] and higher indicates more likely hallucinated.
    """

    def __init__(
        self,
        model_name: str = "mnli-base",
        granularity: str = "sentence",
        device: str = "cpu",
        max_doc_sents: int = 100,
        cache_path: Optional[str] = None,
        imager: Optional[Any] = None
    ) -> None:
        """
        Initialize NLIChecker.

        Args:
            model_name: NLI model backend (default: 'mnli-base', microsoft/deberta-base-mnli).
            granularity: Document chunk granularity ('sentence' or 'paragraph').
            device: 'cpu' or 'cuda'.
            max_doc_sents: Maximum document chunks to score against.
            cache_path: Path to disk cache JSON file.
            imager: Optional pre-initialized or mocked SummaCImager (useful for unit testing).
        """
        self.model_name = model_name
        self.granularity = granularity
        self.device = device
        self.max_doc_sents = max_doc_sents

        if cache_path is None:
            # Default to results/cache_approach_a.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "results"))
            self.cache_path = os.path.join(results_dir, f"cache_approach_a_{model_name.replace('-', '_')}.json")
        else:
            self.cache_path = os.path.abspath(cache_path)

        self._disk_cache: Dict[str, Dict[str, Any]] = self._load_cache()

        if imager is not None:
            self.imager = imager
        elif SummaCImager is not None:
            self.imager = SummaCImager(
                model_name=self.model_name,
                granularity=self.granularity,
                device=self.device,
                max_doc_sents=self.max_doc_sents
            )
        else:
            self.imager = None

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

    @staticmethod
    def support_to_hallucination(support_score: float) -> float:
        """
        Convert theoretical SummaC support score S in [-1.0, +1.0] to hallucination_score in [0.0, 1.0].

        Theoretical Rationale:
          SummaC computes Support S = P(Entailment) - P(Contradiction).
          The theoretical bounds of S are [-1.0, +1.0]:
            +1.0 = Maximum entailment / factual consistency
             0.0 = Neutral / uninformative
            -1.0 = Maximum contradiction / factual inconsistency
          
          Normalising support to [0.0, 1.0]:
            norm_support = (S - (-1.0)) / (1.0 - (-1.0)) = (S + 1.0) / 2.0
          
          Inverting so higher = more likely hallucinated:
            hallucination_score = 1.0 - norm_support = (1.0 - S) / 2.0

          Bounded to [0.0, 1.0] to guard against floating-point precision deviations.
        """
        h_score = (1.0 - support_score) / 2.0
        return float(np.clip(h_score, 0.0, 1.0))

    def score_sentences(
        self,
        judgment_text: str,
        sentences: List[str]
    ) -> List[float]:
        """
        Score a list of summary sentences against the source judgment text.

        Returns:
            List[float]: hallucination scores in [0.0, 1.0] in the same order as sentences.
        """
        if not sentences:
            return []

        doc_hash = compute_text_hash(judgment_text)
        scores: List[float] = [0.0] * len(sentences)
        missing_indices: List[int] = []
        missing_sentences: List[str] = []

        # 1. Retrieve cached entries
        for idx, sent in enumerate(sentences):
            sent_key = f"{doc_hash}:::{sent.strip()}:::{self.model_name}"
            if sent_key in self._disk_cache:
                scores[idx] = self._disk_cache[sent_key]["hallucination_score"]
            else:
                missing_indices.append(idx)
                missing_sentences.append(sent.strip())

        # 2. Score uncached sentences
        if missing_sentences:
            if self.imager is None:
                raise RuntimeError("SummaCImager is not initialized.")

            # Load model lazily if needed
            if hasattr(self.imager, "model") and self.imager.model is None:
                self.imager.load_nli()

            # Split document chunks
            doc_chunks = self.imager.split_text(judgment_text, granularity=self.granularity)[:self.max_doc_sents]
            if not doc_chunks:
                doc_chunks = [judgment_text[:500]]

            # Score each missing sentence
            for local_idx, orig_idx in enumerate(missing_indices):
                sent_text = missing_sentences[local_idx]
                image = self.imager.build_image(judgment_text, sent_text)

                # image shape is (3, N_doc_chunks, N_gen_chunks)
                # Channel 0: Entailment, Channel 1: Contradiction, Channel 2: Neutral
                if image.shape[1] > 0 and image.shape[2] > 0:
                    ent_scores = image[0, :, 0]
                    con_scores = image[1, :, 0]

                    max_ent = float(np.max(ent_scores))
                    max_con = float(np.max(con_scores))
                    best_doc_idx = int(np.argmax(ent_scores))
                    best_doc_chunk = doc_chunks[best_doc_idx] if best_doc_idx < len(doc_chunks) else ""

                    support_score = max_ent - max_con
                else:
                    max_ent = 0.0
                    max_con = 0.0
                    best_doc_idx = -1
                    best_doc_chunk = ""
                    support_score = 0.0

                h_score = self.support_to_hallucination(support_score)
                scores[orig_idx] = h_score

                # Store in disk cache
                sent_key = f"{doc_hash}:::{sent_text}:::{self.model_name}"
                self._disk_cache[sent_key] = {
                    "hallucination_score": round(h_score, 6),
                    "support_score": round(support_score, 6),
                    "max_entailment": round(max_ent, 6),
                    "max_contradiction": round(max_con, 6),
                    "best_doc_idx": best_doc_idx,
                    "best_doc_chunk": best_doc_chunk
                }

            self._save_cache()

        return scores

    def get_sentence_evidence(
        self,
        judgment_text: str,
        sentence: str
    ) -> Dict[str, Any]:
        """Retrieve detailed NLI evidence and matched judgment sentence from cache."""
        doc_hash = compute_text_hash(judgment_text)
        sent_key = f"{doc_hash}:::{sentence.strip()}:::{self.model_name}"
        if sent_key in self._disk_cache:
            return self._disk_cache[sent_key]
        return {}
