"""
Hallucination Detector (hd) package.

A modular toolkit for detecting hallucinations in AI-generated summaries
of Indian legal judgments using NLI and Self-Consistency checking.
"""

from .text_utils import clean_text, split_sentences
from .data_loader import load_mildsum_pairs

__all__ = ["clean_text", "split_sentences", "load_mildsum_pairs"]
__version__ = "0.1.0"
