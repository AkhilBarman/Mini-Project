"""
Unit tests for hd.mutations.

Tests that each mutator alters meaning, returns the expected error_type,
and safely returns None when conditions are not applicable.
"""

import pytest
from hd.mutations import (
    ERROR_TYPES,
    FABRICATED_TEMPLATES,
    get_applicable_mutations,
    mutate_date_number,
    mutate_fabricated,
    mutate_name_place,
    mutate_negation,
    mutate_party_outcome,
    mutate_sentence,
)


def test_mutate_date_number_applies():
    sent = "The High Court directed payment of Rs. 16 lakhs on 20.06.2020 under Section 302."
    res = mutate_date_number(sent, seed=42)
    assert res is not None
    edited, etype, desc = res
    assert etype == "DATE_NUMBER"
    assert edited != sent
    assert len(desc) > 0


def test_mutate_date_number_not_applicable():
    sent = "The learned counsel presented extensive arguments before the division bench."
    res = mutate_date_number(sent, seed=42)
    assert res is None


def test_mutate_party_outcome_applies():
    sent = "The High Court allowed the appeal filed by the appellant."
    res = mutate_party_outcome(sent, seed=42)
    assert res is not None
    edited, etype, desc = res
    assert etype == "PARTY_OUTCOME"
    assert edited != sent
    assert ("dismissed" in edited.lower() or "respondent" in edited.lower())


def test_mutate_party_outcome_not_applicable():
    sent = "The circular was published in the official gazette yesterday morning."
    res = mutate_party_outcome(sent, seed=42)
    assert res is None


def test_mutate_name_place_applies():
    sent = "The Delhi High Court heard the plea in Saket Courts."
    res = mutate_name_place(sent, seed=42)
    assert res is not None
    edited, etype, desc = res
    assert etype == "NAME_PLACE"
    assert edited != sent
    assert ("Bombay High Court" in edited or "Tis Hazari Courts" in edited)


def test_mutate_name_place_not_applicable():
    sent = "The technical standing committee recommended a complete reimbursement."
    res = mutate_name_place(sent, seed=42)
    assert res is None


def test_mutate_negation_applies():
    sent = "The petitioner cannot be simply told that he failed to abide by the circular."
    res = mutate_negation(sent, seed=42)
    assert res is not None
    edited, etype, desc = res
    assert etype == "NEGATION"
    assert edited != sent


def test_mutate_fabricated():
    sent = "Any input sentence here."
    edited, etype, desc = mutate_fabricated(sent, seed=42)
    assert etype == "FABRICATED"
    assert edited in FABRICATED_TEMPLATES
    assert len(desc) > 0


def test_mutate_sentence_dispatches_correctly():
    sent = "The High Court allowed the writ petition."
    res_po = mutate_sentence(sent, target_error="PARTY_OUTCOME", seed=42)
    assert res_po is not None
    assert res_po[1] == "PARTY_OUTCOME"

    res_fab = mutate_sentence(sent, target_error="FABRICATED", seed=42)
    assert res_fab is not None
    assert res_fab[1] == "FABRICATED"

    # Empty sentence check
    assert mutate_sentence("") is None
    assert mutate_sentence(None) is None


def test_get_applicable_mutations_returns_valid_dict():
    sent = "The Delhi High Court allowed the appeal for Rs. 10 lakhs."
    app = get_applicable_mutations(sent, seed=42)
    assert "FABRICATED" in app
    assert "DATE_NUMBER" in app
    assert "PARTY_OUTCOME" in app
    assert "NAME_PLACE" in app
