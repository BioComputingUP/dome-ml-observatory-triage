import math
import pytest
from dome_triage.dedupe.keys import canonical_key_from_ids, record_id_from_ids


def test_float_nan_is_not_treated_as_a_real_id():
    # Real incident: pd.read_csv(dtype=str) yields float('nan') for a missing id, which is TRUTHY,
    # so it produced the key "PMCID:nan" and fused every id-less record into one.
    assert canonical_key_from_ids(float("nan"), float("nan"), float("nan")) is None
    assert record_id_from_ids(float("nan"), float("nan"), float("nan")) is None


def test_literal_nan_string_is_not_treated_as_a_real_id():
    # Same collision via a str() round-trip: str(float('nan')) == "nan".
    assert canonical_key_from_ids("nan", "nan", "nan") is None
    assert record_id_from_ids("nan", "nan", "nan") is None


@pytest.mark.parametrize("junk", ["None", "null", "NA", "n/a", "<NA>", "", "   ", "NaN"])
def test_other_missing_sentinels_are_rejected(junk):
    assert canonical_key_from_ids(junk, junk, junk) is None


def test_nan_pmcid_falls_through_to_the_real_doi_instead_of_winning():
    # The priority order is pmcid -> doi -> pmid; a NaN pmcid must not short-circuit it.
    assert canonical_key_from_ids(float("nan"), None, "10.1/real") == "DOI:10.1/real"


def test_real_ids_are_unchanged_and_still_take_priority_order():
    assert canonical_key_from_ids("PMC1", "123", "10.1/x") == "PMCID:PMC1"
    assert canonical_key_from_ids(None, "123", "10.1/x") == "DOI:10.1/x"
    assert canonical_key_from_ids(None, "123", None) == "PMID:123"


def test_surrounding_whitespace_is_stripped_not_treated_as_a_different_record():
    assert canonical_key_from_ids(" PMC1 ", None, None) == "PMCID:PMC1"


def test_two_records_missing_every_id_never_share_an_id():
    assert record_id_from_ids(None, None, None) is None
    assert record_id_from_ids(float("nan"), None, "") is None
