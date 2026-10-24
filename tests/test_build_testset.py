"""
Unit tests for testset building using isolated mock fixtures.

Never depends on real dataset files.
"""

import csv
import json
import pytest
from scripts.build_testset import build_testset


def test_build_testset_isolated_fixture(tmp_path):
    # 1. Create a mock processed JSONL file
    mock_jsonl = tmp_path / "mock_pairs.jsonl"
    mock_data = [
        {
            "id": "TestDoc_1",
            "judgment_text": "This is a full judgment text.",
            "summary_text": "First sentence. Second sentence. Third sentence.",
            "judgment_sentences": ["This is a full judgment text."],
            "summary_sentences": ["First sentence.", "Second sentence.", "Third sentence."]
        }
    ]
    with open(mock_jsonl, "w", encoding="utf-8") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

    # 2. Create a mock review CSV with 1 accepted and 1 rejected mutation
    mock_review_csv = tmp_path / "mock_review.csv"
    with open(mock_review_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate_id", "pair_id", "sentence_idx", "original_sentence",
            "edited_sentence", "error_type", "accept", "fixed_sentence", "notes"
        ])
        writer.writeheader()
        # Accepted mutation for sentence index 1 (Second sentence)
        writer.writerow({
            "candidate_id": "cand_01",
            "pair_id": "TestDoc_1",
            "sentence_idx": "1",
            "original_sentence": "Second sentence.",
            "edited_sentence": "Second sentence MUTATED.",
            "error_type": "PARTY_OUTCOME",
            "accept": "Y",
            "fixed_sentence": "",
            "notes": "Test mutation"
        })
        # Rejected mutation for sentence index 2 (Third sentence)
        writer.writerow({
            "candidate_id": "cand_02",
            "pair_id": "TestDoc_1",
            "sentence_idx": "2",
            "original_sentence": "Third sentence.",
            "edited_sentence": "Third sentence REJECTED.",
            "error_type": "DATE_NUMBER",
            "accept": "N",
            "fixed_sentence": "",
            "notes": "Rejected by reviewer"
        })

    # 3. Build testset
    output_csv = tmp_path / "output_testset.csv"
    rows = build_testset(
        review_csv_path=str(mock_review_csv),
        pairs_jsonl_path=str(mock_jsonl),
        output_csv_path=str(output_csv)
    )

    # 4. Assertions
    # 3 sentences in doc -> 3 original + 3 edited = 6 total rows
    assert len(rows) == 6

    # Verify original rows (indices 0, 1, 2)
    orig_rows = [r for r in rows if r["version"] == "original"]
    assert len(orig_rows) == 3
    for r in orig_rows:
        assert r["label"] == "0"
        assert r["error_type"] == ""
        assert r["original_sentence"] == ""

    # Verify edited rows
    edited_rows = [r for r in rows if r["version"] == "edited"]
    assert len(edited_rows) == 3

    # Index 0 was unmutated
    assert edited_rows[0]["label"] == "0"
    assert edited_rows[0]["sentence_text"] == "First sentence."

    # Index 1 was mutated and accepted -> label = 1
    assert edited_rows[1]["label"] == "1"
    assert edited_rows[1]["sentence_text"] == "Second sentence MUTATED."
    assert edited_rows[1]["error_type"] == "PARTY_OUTCOME"
    assert edited_rows[1]["original_sentence"] == "Second sentence."

    # Index 2 was rejected (accept = N) -> label = 0
    assert edited_rows[2]["label"] == "0"
    assert edited_rows[2]["sentence_text"] == "Third sentence."
