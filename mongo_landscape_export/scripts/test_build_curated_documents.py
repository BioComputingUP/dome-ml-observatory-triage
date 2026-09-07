"""Tests for the curated -> document conversion logic.

Pure functions only -- nothing here reads the real 1.83GB bulk pool or the real curated set, per
AGENTS.md's rule that fixtures must never depend on the multi-GB inputs being present.
"""

from __future__ import annotations

import json

import pytest

from build_curated_documents import (
    _merge_sources,
    build_rationale,
    normalise,
    sort_key,
)

BULK_TWIN = {
    "pmcid": "PMC8371605", "pmid": "34265844", "doi": "10.1038/s41586-021-03819-2",
    "title": "Highly accurate protein structure prediction with AlphaFold",
    "abstract": "Proteins are essential to life.", "authors": "Jumper J", "year": "2021",
    "journal": "Nature",
    "mesh_headings": '["Deep Learning", "Neural Networks, Computer"]',
    "pub_types": '["Journal Article"]', "keywords_author": "[]",
    "is_open_access": "True", "fulltext_available": "True",
    "abstract_source": "europepmc", "metadata_repair_sources": "",
}


def _canonical(**overrides) -> dict:
    row = {
        "record_id": "r1", "canonical_key": "PMCID:PMC8371605",
        "pmcid": "PMC8371605", "pmid": "34265844", "doi": "10.1038/s41586-021-03819-2",
        "title": "", "abstract": "", "authors": "", "year": "", "journal": "",
        "mesh_headings": "[]", "pub_types": "[]", "keywords_author": "[]",
        "is_open_access": "True", "fulltext_available": "True",
        "abstract_source": "", "metadata_repair_sources": "",
        "label": "positive", "label_confidence": "registry_confirmed",
        "sources": json.dumps([{"source_name": "dome_registry_231_gold", "matched_on": "pmcid"}]),
        "notes": "", "curation_tag": "", "cross_curate_notes": "",
        "cross_curate_final_label": "", "cross_curate_curator": "",
        "original_cohort_review_curator": "", "updated_at": "2026-08-14T17:28:06+00:00",
    }
    row.update(overrides)
    return row


def _by_id(twin: dict = BULK_TWIN) -> dict:
    return {twin["pmcid"]: twin, twin["pmid"]: twin, twin["doi"]: twin}


def _run(rows, timestamps=None, licensing=None, citations=None):
    return normalise(rows, _by_id(), timestamps or {}, licensing or {}, citations or {},
                     "curated_merge_test")


# -- the facet backfill -----------------------------------------------------


def test_facets_are_backfilled_from_the_bulk_pool():
    """Curated rows came from PDF and registry sources, not an EPMC core fetch, so they carry
    empty mesh/pub_types/keywords. Loading 6k records with empty facets would silently distort
    every facet count in the UI."""
    out, stats, _ = _run([_canonical()])
    assert json.loads(out[0]["mesh_headings"]) == ["Deep Learning", "Neural Networks, Computer"]
    assert json.loads(out[0]["pub_types"]) == ["Journal Article"]
    assert stats["backfilled_facets"] == 1


def test_the_bulk_pool_never_overwrites_a_real_curated_value():
    out, _, _ = _run([_canonical(title="A title the curator recorded")])
    assert out[0]["title"] == "A title the curator recorded"


def test_a_row_with_no_bulk_twin_is_still_converted():
    out, stats, _ = normalise([_canonical(pmcid="PMC_UNKNOWN", pmid="", doi="")],
                              {}, {}, {}, {}, "b")
    assert stats["no_bulk_twin"] == 1 and len(out) == 1


# -- dedupe on pid ----------------------------------------------------------


def test_two_canonical_rows_for_one_paper_collapse_to_one_document():
    """AlphaFold 2 is exactly this: one PMCID-keyed row and one DOI-keyed row."""
    rows = [_canonical(record_id="r1", canonical_key="PMCID:PMC8371605"),
            _canonical(record_id="r2", canonical_key="DOI:10.1038/s41586-021-03819-2")]
    out, stats, _ = _run(rows)
    assert len(out) == 1 and stats["deduped_away"] == 1


def test_provenance_is_the_strongest_claim_in_the_group():
    """Picking by timestamp alone lost 162 of the 378 registry_confirmed positives, because those
    papers also carry an agreeing human_curated row written later. A human review agreeing with
    the DOME Registry does not make the registry entry weaker."""
    rows = [
        _canonical(record_id="r1", label_confidence="registry_confirmed",
                   updated_at="2026-01-01T00:00:00+00:00"),
        _canonical(record_id="r2", label_confidence="human_curated",
                   updated_at="2026-08-01T00:00:00+00:00"),
    ]
    out, stats, _ = _run(rows)
    assert out[0]["label_confidence"] == "registry_confirmed"
    assert stats["provenance_upgraded"] == 1


def test_the_rationale_names_every_source_that_agreed():
    rows = [
        _canonical(record_id="r1",
                   sources=json.dumps([{"source_name": "dome_registry_231_gold",
                                        "matched_on": "pmcid"}])),
        _canonical(record_id="r2",
                   sources=json.dumps([{"source_name": "ebi_search_dome_api",
                                        "matched_on": "doi"}])),
    ]
    out, _, _ = _run(rows)
    assert "dome_registry_231_gold" in out[0]["curation_rationale"]
    assert "ebi_search_dome_api" in out[0]["curation_rationale"]


# -- conflicts --------------------------------------------------------------


def test_a_human_review_overrides_a_registry_entry():
    """Gavin's rule (2026-09-03): human judgement is retained. A curator who read the paper and
    recorded a contrary verdict has deliberately overridden the registry."""
    rows = [
        _canonical(record_id="r1", label="positive", label_confidence="registry_confirmed"),
        _canonical(record_id="r2", label="negative", label_confidence="human_curated",
                   cross_curate_notes="Not unsupervised ML - classical clustering"),
    ]
    out, stats, conflicts = _run(rows)
    assert len(out) == 1
    assert out[0]["label"] == "negative"
    assert stats["conflicts_resolved"] == 1


def test_an_overridden_paper_is_not_published_as_registry_confirmed():
    """Claiming registry confirmation for a verdict that contradicts the registry would be false."""
    rows = [
        _canonical(record_id="r1", label="positive", label_confidence="registry_confirmed"),
        _canonical(record_id="r2", label="negative", label_confidence="human_curated"),
    ]
    out, _, _ = _run(rows)
    assert out[0]["label_confidence"] == "human_curated"


def test_between_two_human_decisions_the_most_recent_wins():
    rows = [_canonical(record_id="r1", label="negative"),
            _canonical(record_id="r2", label="positive")]
    out, _, _ = _run(rows, timestamps={"r1": "2026-01-01T00:00:00+00:00",
                                       "r2": "2026-08-26T21:34:30+00:00"})
    assert out[0]["label"] == "positive"


def test_every_conflicting_row_is_audited_with_its_outcome():
    rows = [_canonical(record_id="r1", label="positive", label_confidence="registry_confirmed"),
            _canonical(record_id="r2", label="negative", label_confidence="human_curated")]
    _, _, conflicts = _run(rows)
    assert len(conflicts) == 2
    published = [c for c in conflicts if c["resolution"] == "PUBLISHED"]
    assert len(published) == 1 and published[0]["label"] == "negative"
    assert "overrode the registry" in published[0]["resolution_reason"]
    assert all(c["resolution_reason"] for c in conflicts)


def test_agreeing_rows_are_not_treated_as_a_conflict():
    out, stats, conflicts = _run([_canonical(record_id="r1"), _canonical(record_id="r2")])
    assert len(out) == 1 and stats["conflicts_resolved"] == 0 and conflicts == []


def test_agreeing_rows_still_keep_the_strongest_provenance():
    """The inverse rule, and both are right: when rows AGREE, a registry entry a human also
    confirmed is still a registry entry. Only a disagreement demotes it."""
    rows = [_canonical(record_id="r1", label="positive", label_confidence="registry_confirmed",
                       updated_at="2026-01-01T00:00:00+00:00"),
            _canonical(record_id="r2", label="positive", label_confidence="human_curated",
                       updated_at="2026-08-01T00:00:00+00:00")]
    out, _, _ = _run(rows)
    assert out[0]["label_confidence"] == "registry_confirmed"


# -- what is and is not published -------------------------------------------


def test_skipped_records_are_dropped():
    """The schema enum has no 'skipped'; a curator declining to judge is not a verdict."""
    out, stats, _ = _run([_canonical(label="skipped")])
    assert out == [] and stats["dropped_label"] == 1


def test_undeterminable_records_are_kept():
    out, _, _ = _run([_canonical(label="undeterminable", label_confidence="human_curated")])
    assert len(out) == 1 and out[0]["label"] == "undeterminable"


def test_heuristic_candidate_rows_are_never_published():
    """The clear-negative sampler rows were fetched with the structural inverse of the AI/ML
    query and were never in scope for this corpus."""
    out, stats, _ = _run([_canonical(label="negative", label_confidence="heuristic_candidate")])
    assert out == [] and stats["dropped_confidence"] == 1


# -- joins ------------------------------------------------------------------


def test_licence_join_marks_checked_versus_never_looked_up():
    with_licence, _, _ = _run([_canonical()], licensing={"34265844": ("cc by", "Y")})
    assert with_licence[0]["license"] == "cc by"
    assert with_licence[0]["license_checked"] == "True"

    without, _, _ = _run([_canonical()], licensing={})
    assert without[0]["license"] == "" and without[0]["license_checked"] == "False"


def test_citations_are_found_by_identifier_not_by_pid():
    """These records were absent from the landscape corpus, so they have no row in the
    pid-keyed mapping -- their counts have to be looked up by identifier."""
    citations = {("pmid", "34265844"): ("34984", "2026-09-03T00:00:00+00:00")}
    out, stats, _ = _run([_canonical()], citations=citations)
    assert out[0]["citation_count"] == "34984"
    assert out[0]["citation_source"] == "europepmc"
    assert stats["citation_matched"] == 1


def test_a_record_with_no_citation_gets_blank_fields_not_a_zero():
    out, _, _ = _run([_canonical()], citations={})
    assert out[0]["citation_count"] == "" and out[0]["citation_source"] == ""


# -- rationale + helpers ----------------------------------------------------


def test_rationale_distinguishes_registry_from_human():
    assert build_rationale(_canonical()).startswith("Confirmed DOME Registry entry")
    assert build_rationale(_canonical(label_confidence="human_curated")).startswith(
        "Human-curated decision")


def test_rationale_carries_a_real_note_rather_than_inventing_one():
    text = build_rationale(_canonical(cross_curate_notes="Disagree - clear ML use in abstract"))
    assert "Disagree - clear ML use in abstract" in text


def test_rationale_survives_malformed_sources_json():
    assert build_rationale(_canonical(sources="{not json")).endswith(".")


def test_merge_sources_deduplicates_on_name_and_match_key():
    merged = json.loads(_merge_sources([
        _canonical(sources=json.dumps([{"source_name": "a", "matched_on": "pmcid"}])),
        _canonical(sources=json.dumps([{"source_name": "a", "matched_on": "pmcid"},
                                       {"source_name": "b", "matched_on": "doi"}])),
    ]))
    assert sorted(s["source_name"] for s in merged) == ["a", "b"]


def test_sort_key_prefers_a_real_curation_event_over_updated_at():
    row = _canonical(updated_at="2020-01-01T00:00:00+00:00")
    assert sort_key(row, {"r1": "2026-08-26T21:34:30+00:00"})[0] == "2026-08-26T21:34:30+00:00"
    assert sort_key(row, {})[0] == "2020-01-01T00:00:00+00:00"
