from __future__ import annotations

import uuid

import pytest

from pid import landscape_identity_key, mint_landscape_pid


def test_deterministic_same_inputs_same_pid():
    a = mint_landscape_pid("PMC123", "10.1/x", "999")
    b = mint_landscape_pid("PMC123", "10.1/x", "999")
    assert a == b


def test_different_papers_different_pids():
    a = mint_landscape_pid("PMC123", None, None)
    b = mint_landscape_pid("PMC456", None, None)
    assert a != b


def test_priority_pmcid_beats_doi_beats_pmid():
    # Same pmcid, different doi/pmid -> identical PID, since pmcid wins the priority order.
    a = mint_landscape_pid("PMC123", "10.1/one", "111")
    b = mint_landscape_pid("PMC123", "10.1/two", "222")
    assert a == b

    # No pmcid -> falls through to doi.
    c = mint_landscape_pid(None, "10.1/one", "111")
    d = mint_landscape_pid(None, "10.1/one", "222")
    assert c == d
    assert c != a


def test_blank_and_nan_like_strings_are_skipped_not_treated_as_ids():
    # Mirrors dedupe/keys.py's real bug history: a pandas-NaN-turned-string must not be treated as
    # a usable id.
    a = mint_landscape_pid("nan", "10.1/real", "999")
    b = mint_landscape_pid("", "10.1/real", "999")
    c = mint_landscape_pid(None, "10.1/real", "999")
    assert a == b == c


def test_no_usable_id_raises():
    assert landscape_identity_key("", "nan", None) is None
    with pytest.raises(ValueError):
        mint_landscape_pid("", "nan", None)


def test_pid_is_a_well_formed_uuid_string():
    pid = mint_landscape_pid("PMC123", None, None)
    assert uuid.UUID(pid) is not None
