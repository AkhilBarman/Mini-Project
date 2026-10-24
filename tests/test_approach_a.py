"""
Tests for Approach A (NLIChecker using SummaC).

Includes:
- Fast unit tests using mocked SummaCImager (no heavy downloads or model loading).
- Verification of theoretical support-to-hallucination score conversion.
- Verification of persistent disk caching.
- Slow integration test marked @pytest.mark.slow (run with pytest -m slow).
"""

import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from hd.approach_a import NLIChecker


def test_support_to_hallucination_theoretical_range():
    """Verify that theoretical support bounds map to hallucination bounds."""
    # S = +1.0 (Maximum Support / Entailment) -> H = 0.0 (Not Hallucinated)
    assert NLIChecker.support_to_hallucination(1.0) == 0.0

    # S = -1.0 (Maximum Contradiction) -> H = 1.0 (Completely Hallucinated)
    assert NLIChecker.support_to_hallucination(-1.0) == 1.0

    # S = 0.0 (Neutral / Uninformative) -> H = 0.5
    assert NLIChecker.support_to_hallucination(0.0) == 0.5

    # Intermediate values
    assert abs(NLIChecker.support_to_hallucination(0.5) - 0.25) < 1e-6
    assert abs(NLIChecker.support_to_hallucination(-0.5) - 0.75) < 1e-6

    # Boundary protection / clipping
    assert NLIChecker.support_to_hallucination(2.0) == 0.0
    assert NLIChecker.support_to_hallucination(-2.0) == 1.0


def test_score_sentences_empty_input(tmp_path):
    """Empty list of sentences returns empty list without error."""
    cache_file = tmp_path / "cache.json"
    checker = NLIChecker(cache_path=str(cache_file))
    res = checker.score_sentences("Sample judgment text.", [])
    assert res == []


def test_score_sentences_mocked(tmp_path):
    """Fast test using a mock imager to verify score calculation and ranking."""
    mock_imager = MagicMock()
    mock_imager.split_text.return_value = ["Doc chunk 1", "Doc chunk 2"]

    # Mock build_image:
    # First sentence (supported): high entailment (0.9), low contradiction (0.05)
    # Second sentence (hallucinated): low entailment (0.05), high contradiction (0.85)
    def fake_build_image(doc, sent):
        img = np.zeros((3, 2, 1))
        if "supported" in sent.lower():
            img[0, 0, 0] = 0.90  # Entailment
            img[1, 0, 0] = 0.05  # Contradiction
            img[2, 0, 0] = 0.05  # Neutral
        else:
            img[0, 0, 0] = 0.05  # Entailment
            img[1, 0, 0] = 0.85  # Contradiction
            img[2, 0, 0] = 0.10  # Neutral
        return img

    mock_imager.build_image.side_effect = fake_build_image

    cache_file = tmp_path / "cache.json"
    checker = NLIChecker(
        model_name="mnli-base",
        cache_path=str(cache_file),
        imager=mock_imager
    )

    sentences = [
        "This sentence is supported by the court.",
        "This sentence is hallucinated and fabricated."
    ]

    scores = checker.score_sentences("Judgment text here.", sentences)

    assert len(scores) == 2
    # Supported sentence: S = 0.90 - 0.05 = 0.85 -> H = (1 - 0.85)/2 = 0.075
    # Hallucinated sentence: S = 0.05 - 0.85 = -0.80 -> H = (1 - (-0.80))/2 = 0.90
    assert scores[0] < scores[1]
    assert scores[0] < 0.20
    assert scores[1] > 0.80


def test_caching_mechanism(tmp_path):
    """Verify that second call is served from disk cache without model re-inference."""
    mock_imager = MagicMock()
    mock_imager.split_text.return_value = ["Judgment chunk"]
    mock_imager.build_image.return_value = np.zeros((3, 1, 1))

    cache_file = tmp_path / "test_cache.json"
    checker = NLIChecker(
        model_name="mnli-base",
        cache_path=str(cache_file),
        imager=mock_imager
    )

    judgment = "Judgment regarding commercial arbitration."
    sentences = ["The court appointed a sole arbitrator."]

    # First call: invokes imager.build_image once
    scores_1 = checker.score_sentences(judgment, sentences)
    assert mock_imager.build_image.call_count == 1
    assert os.path.exists(str(cache_file))

    # Second call on new instance with same cache file: served from cache
    checker_2 = NLIChecker(
        model_name="mnli-base",
        cache_path=str(cache_file),
        imager=mock_imager
    )
    scores_2 = checker_2.score_sentences(judgment, sentences)

    assert scores_1 == scores_2
    # build_image call count should NOT have increased
    assert mock_imager.build_image.call_count == 1


@pytest.mark.slow
def test_real_model_supported_vs_contradicted(tmp_path):
    """
    Integration test using the real deberta-base-mnli model.
    Checks that a supported sentence scores lower than a contradicted sentence.
    """
    toy_judgment = (
        "The High Court of Delhi on Monday dismissed the appeal filed by the applicant. "
        "The court held that no ground for interference was made out."
    )
    supported_sent = "The Delhi High Court dismissed the appeal filed by the applicant."
    contradicted_sent = "The Delhi High Court allowed the appeal and granted full relief."

    cache_file = tmp_path / "slow_cache.json"
    checker = NLIChecker(
        model_name="mnli-base",
        device="cpu",
        cache_path=str(cache_file)
    )

    scores = checker.score_sentences(toy_judgment, [supported_sent, contradicted_sent])
    h_supp, h_cont = scores[0], scores[1]

    # Hallucination score for contradicted statement should be strictly higher than supported
    assert h_supp < h_cont
