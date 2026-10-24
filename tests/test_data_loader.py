"""
Unit tests for hd.data_loader using isolated temporary fixtures.

Never touches or depends on the real MILDSum dataset files.
"""

import os
import pytest
from hd.data_loader import load_mildsum_pairs, read_text_file


def test_load_valid_fixtures(tmp_path):
    # Setup 2 valid sample folders
    s1 = tmp_path / "Sample_1"
    s1.mkdir()
    (s1 / "EN_Judgment.txt").write_text(
        "1. This is a full judgment text regarding a legal dispute between two parties.",
        encoding="utf-8"
    )
    (s1 / "EN_Summary.txt").write_text(
        "The High Court held that the petition was maintainable.",
        encoding="utf-8"
    )

    s2 = tmp_path / "Sample_2"
    s2.mkdir()
    (s2 / "EN_Judgment.txt").write_text(
        "2. The second case pertains to an appeal filed against the lower trial court order.",
        encoding="utf-8"
    )
    (s2 / "EN_Summary.txt").write_text(
        "The Supreme Court dismissed the criminal appeal.",
        encoding="utf-8"
    )

    pairs = load_mildsum_pairs(str(tmp_path), verbose=False)
    assert len(pairs) == 2
    assert pairs[0]["id"] == "Sample_1"
    assert "dispute" in pairs[0]["judgment_text"]
    assert "maintainable" in pairs[0]["summary_text"]
    assert pairs[1]["id"] == "Sample_2"


def test_load_missing_files_reported(tmp_path):
    # Missing summary file
    s_incomplete = tmp_path / "Sample_Incomplete"
    s_incomplete.mkdir()
    (s_incomplete / "EN_Judgment.txt").write_text(
        "This judgment exists but summary does not exist in folder.",
        encoding="utf-8"
    )

    pairs = load_mildsum_pairs(str(tmp_path), verbose=False)
    assert len(pairs) == 0


def test_load_skips_empty_files(tmp_path):
    s_empty = tmp_path / "Sample_Empty"
    s_empty.mkdir()
    (s_empty / "EN_Judgment.txt").write_text("Short", encoding="utf-8")
    (s_empty / "EN_Summary.txt").write_text("Also short", encoding="utf-8")

    pairs = load_mildsum_pairs(str(tmp_path), min_char_length=30, verbose=False)
    assert len(pairs) == 0


def test_load_handles_encoding(tmp_path):
    s_enc = tmp_path / "Sample_Encoding"
    s_enc.mkdir()
    # Write CP1252 text with smart quotes and Latin-1 with accented characters
    special_text = "Judgment with special quote: ‘State vs. Accused’ and symbol © 2021."
    (s_enc / "EN_Judgment.txt").write_bytes(special_text.encode("cp1252"))
    (s_enc / "EN_Summary.txt").write_bytes("Summary with café and résumé.".encode("latin-1"))

    pairs = load_mildsum_pairs(str(tmp_path), min_char_length=10, verbose=False)
    assert len(pairs) == 1
    assert "Accused" in pairs[0]["judgment_text"]
    assert "café" in pairs[0]["summary_text"] or "caf" in pairs[0]["summary_text"]


def test_load_nonexistent_directory():
    pairs = load_mildsum_pairs("nonexistent/path/here", verbose=False)
    assert pairs == []
