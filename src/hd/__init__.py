"""
Hallucination Detector (hd) package.

A modular toolkit for detecting hallucinations in AI-generated summaries
of Indian legal judgments using NLI and Self-Consistency checking.
"""

from .text_utils import clean_text, split_sentences
from .data_loader import load_mildsum_pairs
from .mutations import ERROR_TYPES, mutate_sentence, get_applicable_mutations
from .evaluation import (
    Timer,
    compute_metrics,
    best_f1_threshold,
    per_error_type_recall,
    save_results,
    make_comparison_table
)
from .approach_a import NLIChecker

__all__ = [
    "clean_text",
    "split_sentences",
    "load_mildsum_pairs",
    "ERROR_TYPES",
    "mutate_sentence",
    "get_applicable_mutations",
    "Timer",
    "compute_metrics",
    "best_f1_threshold",
    "per_error_type_recall",
    "save_results",
    "make_comparison_table",
    "NLIChecker"
]
__version__ = "0.3.0"
