import hashlib

from dome_triage.dedupe.keys import (
    build_clusters,
    canonical_key_from_ids,
    choose_canonical_key,
    record_id_from_canonical_key,
    record_id_from_ids,
)
from dome_triage.schema import RawRecord


def _record(**kwargs) -> RawRecord:
    defaults = {
        "source_name": "test",
        "source_file": "test.csv",
        "label": "positive",
        "label_confidence": "human_curated",
    }
    defaults.update(kwargs)
    return RawRecord(**defaults)


def test_records_sharing_only_a_doi_are_merged():
    # Record A has no DOI overlap partner via PMCID/PMID -- only the DOI connects them.
    a = _record(doi="10.1000/shared", pmcid="PMC1000001")
    b = _record(doi="10.1000/shared", pmid="22222222")

    clusters = build_clusters([a, b])

    assert len(clusters) == 1
    assert len(clusters[0]) == 2
    assert a in clusters[0] and b in clusters[0]


def test_records_with_different_pmcids_are_not_merged_even_if_similar():
    a = _record(pmcid="PMC1000001", title="A machine learning paper")
    b = _record(pmcid="PMC1000002", title="A machine learning paper")  # similar title, no fuzzy match

    clusters = build_clusters([a, b])

    assert len(clusters) == 2


def test_transitive_merge_across_three_sources_via_different_shared_ids():
    # A-B share a PMCID; B-C share a PMID; no direct ID in common between A and C.
    a = _record(pmcid="PMC9999999", doi="10.1000/a")
    b = _record(pmcid="PMC9999999", pmid="33333333")
    c = _record(pmid="33333333", doi="10.1000/c")

    clusters = build_clusters([a, b, c])

    assert len(clusters) == 1
    assert len(clusters[0]) == 3
    assert a in clusters[0] and b in clusters[0] and c in clusters[0]


def test_choose_canonical_key_respects_priority_order():
    cluster = [
        _record(pmid="12345678"),
        _record(doi="10.1000/x", pmid="12345678"),
        _record(pmcid="PMC1234567", doi="10.1000/x", pmid="12345678"),
    ]

    assert choose_canonical_key(cluster, ("pmcid", "doi", "pmid")) == "PMCID:PMC1234567"
    assert choose_canonical_key(cluster, ("doi", "pmcid", "pmid")) == "DOI:10.1000/x"


def test_canonical_key_from_ids_matches_choose_canonical_key_for_the_same_ids():
    # The whole point of canonical_key_from_ids is that a caller with only raw scalar ids (no
    # RawRecord/cluster) gets the identical key a real dedupe consolidate run would produce --
    # verified here by comparing directly against choose_canonical_key on an equivalent cluster.
    cluster = [_record(pmcid="PMC1234567", doi="10.1000/x", pmid="12345678")]

    assert canonical_key_from_ids("PMC1234567", "12345678", "10.1000/x") == choose_canonical_key(cluster)


def test_canonical_key_from_ids_respects_priority_order():
    assert canonical_key_from_ids(None, "12345678", "10.1000/x") == "DOI:10.1000/x"
    assert canonical_key_from_ids(None, "12345678", None) == "PMID:12345678"


def test_canonical_key_from_ids_returns_none_when_no_id_present():
    assert canonical_key_from_ids(None, None, None) is None
    assert canonical_key_from_ids("", "", "") is None


def test_record_id_from_canonical_key_is_plain_sha1():
    key = "PMCID:PMC1234567"
    assert record_id_from_canonical_key(key) == hashlib.sha1(key.encode()).hexdigest()


def test_record_id_from_ids_matches_consolidate_pipeline_output():
    # This is the property the bulk-pool curation feature depends on: a record reached via
    # curate/bulk_pool.py must get the *same* record_id a real dedupe.consolidate run would
    # produce for it, so a decision logged against one reconciles with the other.
    from dome_triage.dedupe.consolidate import _merge_cluster

    record = _record(pmcid="PMC1234567", doi="10.1000/x", pmid="12345678")
    merged = _merge_cluster([record], id_priority=("pmcid", "doi", "pmid"))

    assert record_id_from_ids("PMC1234567", "12345678", "10.1000/x") == merged.record_id


def test_record_id_from_ids_returns_none_when_no_id_present():
    assert record_id_from_ids(None, None, None) is None
