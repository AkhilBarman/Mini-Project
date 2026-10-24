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
from .sampling import LLMSampler, save_sampled_summaries, load_sampled_summaries
from .approach_b import SelfCheckChecker
from .comparison import (
    calibrate_score,
    compute_hybrid_scores,
    build_comparison_dataframe,
    build_error_breakdown_dataframe,
    plot_comparison_dashboard,
    dataframe_to_markdown
)

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
    "NLIChecker",
    "LLMSampler",
    "save_sampled_summaries",
    "load_sampled_summaries",
    "SelfCheckChecker",
    "calibrate_score",
    "compute_hybrid_scores",
    "build_comparison_dataframe",
    "build_error_breakdown_dataframe",
    "plot_comparison_dashboard",
    "dataframe_to_markdown"
]
__version__ = "0.5.0"
