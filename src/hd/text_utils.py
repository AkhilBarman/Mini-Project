"""
Text utilities for Indian legal document processing.

Provides:
- clean_text: Normalises whitespace only, preserving all words, numbers, and dates.
- split_sentences: Robust sentence splitter supporting legal abbreviations and currency.
"""

import re
from typing import List, Optional

# Pre-defined Indian legal abbreviations for sentence boundary disambiguation
INDIAN_LEGAL_ABBREVIATIONS = {
    # Judicial & Legal Titles
    "ors", "anr", "vs", "v", "adv", "hon", "hon'ble", "j", "jj", "c.j", "cj",
    "adj", "dj", "acj", "sc", "hc", "gnctd", "u/s", "u/sec",
    
    # Statutory & Citation terms
    "sec", "secs", "art", "arts", "no", "nos", "para", "paras", "cl", "cls",
    "sub-sec", "sub-s", "reg", "vol", "supp", "ed", "ord",
    
    # Common honorifics & general abbreviations
    "mr", "mrs", "ms", "dr", "prof", "sh", "smt", "km",
    "co", "ltd", "corp", "inc", "pvt", "govt", "dept",
    "i.e", "e.g", "etc", "viz", "al", "ibid", "id",
    
    # Indian Currency
    "rs", "inr"
}


def clean_text(text: Optional[str]) -> str:
    """Normalise whitespace in text.
    
    NEVER alters wording, casing, numbers, dates, names, or legal citations.
    Only normalises irregular spaces (such as non-breaking spaces), converts
    Windows line endings to Unix line endings, and collapses multiple horizontal
    spaces into a single space while keeping logical line structure intact.

    Args:
        text: Input string to clean. If None or non-string, returns empty string.

    Returns:
        Cleaned text string with normalised whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Replace non-breaking spaces and special unicode spaces with standard space
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Standardise line endings to \n
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Normalise horizontal whitespace (spaces, tabs) within each line
    lines = []
    for line in cleaned.split("\n"):
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line_clean)

    # Collapse more than two consecutive empty lines to two
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _get_nltk_legal_tokenizer():
    """Lazily configure and return an NLTK PunktSentenceTokenizer with legal abbreviations."""
    try:
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
    except ImportError:
        return None

    punkt_params = PunktParameters()
    punkt_params.abbrev_types = INDIAN_LEGAL_ABBREVIATIONS
    return PunktSentenceTokenizer(punkt_params)


# Module-level cache for custom tokenizer
_NLTK_LEGAL_TOKENIZER = None


def split_sentences(text: Optional[str], method: str = "nltk_legal") -> List[str]:
    """Split text into sentences while respecting legal abbreviations and currency.

    Handles abbreviations like 'Ors.', 'vs.', 'No.', 'Sec.', 'Art.', 'Mr.',
    'Dr.', 'J.', and currency expressions like 'Rs. 5,000' or 'Rs. 24,02,380'
    without incorrect sentence splits.

    Args:
        text: Input text string.
        method: Splitting engine to use ('nltk_legal' or 'pysbd').
                Defaults to 'nltk_legal'.

    Returns:
        List of non-empty sentence strings.
    """
    if not text:
        return []
    
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if method == "pysbd":
        try:
            import pysbd
            seg = pysbd.Segmenter(language="en", clean=False)
            raw_sentences = seg.segment(cleaned)
            return [s.strip() for s in raw_sentences if s and s.strip()]
        except ImportError:
            # Fall back to nltk_legal if pysbd is not installed
            pass

    # Default: Custom NLTK Legal Punkt Tokenizer
    global _NLTK_LEGAL_TOKENIZER
    if _NLTK_LEGAL_TOKENIZER is None:
        _NLTK_LEGAL_TOKENIZER = _get_nltk_legal_tokenizer()

    if _NLTK_LEGAL_TOKENIZER is not None:
        # Pre-process numbered list items (e.g. "1. The petitioner...")
        # so Punkt doesn't isolate "1." into a separate sentence.
        placeholder_pattern = re.compile(r"(^|\s)(\d+)\.\s+([A-Z])")
        def _preserve_num(m):
            prefix = m.group(1)
            num = m.group(2)
            char = m.group(3)
            return f"{prefix}PARAGRAPH_NUM_{num}_{char}"

        modified_text = placeholder_pattern.sub(_preserve_num, cleaned)
        raw_sentences = _NLTK_LEGAL_TOKENIZER.tokenize(modified_text)
        
        # Restore placeholders
        restore_pattern = re.compile(r"PARAGRAPH_NUM_(\d+)_([A-Z])")
        restored = []
        for s in raw_sentences:
            s_restored = restore_pattern.sub(r"\1. \2", s).strip()
            if s_restored:
                restored.append(s_restored)
        return restored

    # Fallback if NLTK is unavailable: regex-based sentence boundary detection
    # that skips known legal abbreviations and Rs. currency
    abbrev_pattern = r"(?:Rs|Mr|Mrs|Ms|Dr|Sec|Art|No|Nos|vs|v|Ors|Anr|Hon|Adv|Govt|Ltd|Co)\."
    safe_text = re.sub(
        r"\b(" + abbrev_pattern + r")",
        lambda m: m.group(1).replace(".", "__DOT__"),
        cleaned,
        flags=re.IGNORECASE
    )
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", safe_text)
    return [s.replace("__DOT__", ".").strip() for s in raw_sents if s.strip()]
