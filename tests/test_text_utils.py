"""
Unit tests for hd.text_utils.

Tests whitespace normalisation and sentence boundary detection with a focus
on legal abbreviations, currency (Rs. 5,000), and edge cases.
"""

import pytest
from hd.text_utils import clean_text, split_sentences


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_clean_text_preserves_content():
    sample = (
        "The   appellant   under   Sec.  302  IPC was fined Rs. 5,000 on 20.06.2020\n\n"
        "by   Justice   Sharma,\tJ.   in   Case No. 42/2021."
    )
    cleaned = clean_text(sample)
    # Check words, numbers, citations, dates are preserved
    assert "Sec. 302 IPC" in cleaned
    assert "Rs. 5,000" in cleaned
    assert "20.06.2020" in cleaned
    assert "Justice Sharma, J." in cleaned
    assert "Case No. 42/2021" in cleaned
    # Ensure redundant multiple spaces were collapsed
    assert "   " not in cleaned
    assert "\t" not in cleaned


def test_split_sentences_empty_and_none():
    assert split_sentences("") == []
    assert split_sentences(None) == []
    assert split_sentences("   \n\t   ") == []


def test_split_sentences_currency_rs_not_split():
    text = "The High Court ordered compensation of Rs. 5,000 to the victim. The respondent deposited the sum."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Rs. 5,000" in sentences[0]
    assert sentences[1] == "The respondent deposited the sum."


def test_split_sentences_currency_large_amount():
    text = "The hospital demanded Rs. 24,02,380 from the petitioner. Only Rs. 7,08,500 was reimbursed."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Rs. 24,02,380" in sentences[0]
    assert "Rs. 7,08,500" in sentences[1]


def test_split_sentences_legal_abbreviations():
    text = (
        "In ABC & Ors. vs. State of Maharashtra & Anr., the court interpreted Sec. 302 IPC. "
        "The appeal was subsequently dismissed under Art. 136."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Ors. vs. State of Maharashtra & Anr." in sentences[0]
    assert "Sec. 302 IPC." in sentences[0]
    assert "Art. 136." in sentences[1]


def test_split_sentences_titles_and_judges():
    text = "Mr. Justice Dinesh Kumar, J. delivered the judgment. The order takes effect today."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Mr. Justice Dinesh Kumar, J." in sentences[0]
    assert sentences[1] == "The order takes effect today."


def test_split_sentences_numbered_paragraphs():
    text = "1. The petitioner approached the court under Art. 226. 2. The respondent filed a reply."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("1. The petitioner")
    assert sentences[1].startswith("2. The respondent")


def test_split_sentences_pysbd_method():
    text = "The court imposed a penalty of Rs. 10,000 on the contemnor. He failed to appear."
    sentences = split_sentences(text, method="pysbd")
    assert len(sentences) == 2
    assert "Rs. 10,000" in sentences[0]
