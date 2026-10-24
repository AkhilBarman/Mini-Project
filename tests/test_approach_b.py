"""
Unit tests for Phase 4: Approach B (SelfCheckGPT consistency checking and LLM sampling).
"""

import os
from unittest.mock import MagicMock
import pytest

from hd.sampling import LLMSampler, save_sampled_summaries, load_sampled_summaries
from hd.approach_b import SelfCheckChecker, compute_token_containment


def test_sampling_mock():
    """Verify LLMSampler mock backend returns the requested number of non-empty samples."""
    sampler = LLMSampler(backend="mock")
    judgment = "The High Court of Delhi heard the dispute regarding statutory compliance."
    samples = sampler.sample_summaries(judgment, n_samples=3)
    assert len(samples) == 3
    for s in samples:
        assert isinstance(s, str)
        assert len(s) > 10


def test_save_and_load_sampled_summaries(tmp_path):
    """Verify serialization and deserialization of sampled summaries."""
    samples = ["Sample summary 1 about contract law.", "Sample summary 2 about damages."]
    pair_id = "Test_Pair_123"
    
    saved_path = save_sampled_summaries(pair_id, samples, output_dir=str(tmp_path))
    assert os.path.exists(saved_path)
    
    loaded = load_sampled_summaries(pair_id, input_dir=str(tmp_path))
    assert loaded == samples


def test_load_nonexistent_sampled_summaries(tmp_path):
    """Loading a non-existent pair returns an empty list without error."""
    loaded = load_sampled_summaries("nonexistent_id", input_dir=str(tmp_path))
    assert loaded == []


def test_compute_token_containment():
    """Verify directional token containment recall edge cases and range."""
    # Exact match
    sim_exact = compute_token_containment("The court dismissed the appeal.", "The court dismissed the appeal.")
    assert sim_exact == 1.0

    # Completely disjoint
    sim_disjoint = compute_token_containment("apple orange banana", "car truck bicycle")
    assert sim_disjoint == 0.0

    # Partial containment
    sim_partial = compute_token_containment("The High Court dismissed the petition.", "The High Court granted relief.")
    assert 0.0 < sim_partial < 1.0

    # Empty inputs
    assert compute_token_containment("", "text") == 0.0
    assert compute_token_containment("text", "") == 0.0


def test_selfcheck_empty_inputs(tmp_path):
    """Empty sentences or empty samples are handled gracefully."""
    cache_file = tmp_path / "cache.json"
    checker = SelfCheckChecker(cache_path=str(cache_file))
    
    # Empty sentences -> empty result
    assert checker.score_sentences([], ["Sample text."]) == []
    
    # Empty samples -> default neutral 0.5
    scores = checker.score_sentences(["A sentence."], [])
    assert scores == [0.5]


def test_selfcheck_supported_vs_fabricated(tmp_path):
    """
    Supported sentence (present in samples) should receive a low hallucination score.
    Fabricated sentence (absent from samples) should receive a high hallucination score.
    """
    cache_file = tmp_path / "cache.json"
    checker = SelfCheckChecker(method="bertscore", cache_path=str(cache_file))
    
    sampled_summaries = [
        "The High Court of Delhi directed the government to reimburse COVID medical expenses. The petitioner survived after ventilator treatment.",
        "The Delhi High Court allowed the judicial officer's petition for COVID-19 medical reimbursement from the authorities.",
        "Medical reimbursement was ordered by Justice Rekha Palli for COVID treatment incurred at the private hospital."
    ]
    
    supported_sent = "The High Court directed the government to reimburse the medical expenses."
    fabricated_sent = "The petitioner was sentenced to five years rigorous imprisonment in Tihar Jail."
    
    scores = checker.score_sentences([supported_sent, fabricated_sent], sampled_summaries, pair_id="Sample_Test")
    h_supp, h_fab = scores[0], scores[1]
    
    # Fabricated statement should have higher hallucination score than supported
    assert h_fab > h_supp
    assert h_supp < 0.50
    assert h_fab > 0.70


def test_selfcheck_caching_mechanism(tmp_path):
    """Verify that second call uses disk cache without re-invoking scoring."""
    cache_file = tmp_path / "cache.json"
    mock_scorer = MagicMock(return_value=0.85)
    
    checker_1 = SelfCheckChecker(cache_path=str(cache_file), custom_scorer=mock_scorer)
    sent = "Arbitration petition was dismissed."
    samples = ["Sample summary text."]
    
    scores_1 = checker_1.score_sentences([sent], samples, pair_id="P1")
    assert mock_scorer.call_count == 1
    
    # Second call on new instance with same cache
    checker_2 = SelfCheckChecker(cache_path=str(cache_file), custom_scorer=mock_scorer)
    scores_2 = checker_2.score_sentences([sent], samples, pair_id="P1")
    
    assert scores_1 == scores_2
    # mock_scorer should NOT have been called again
    assert mock_scorer.call_count == 1
