"""Tests for the staged-batch -> documents converter."""

from __future__ import annotations

import csv
import json

import pytest

from build_staged_documents import build_row, load_classifications, run
from schema import SCHEMA_VERSION, build_document

STAGED = {
    "pid": "a2321c32-4f26-5098-8038-b26a44a1c3f4",
    "pmid": "34265844", "pmcid": "PMC8371605", "doi": "10.1038/s41586-021-03819-2",
    "title": "Highly accurate protein structure prediction with AlphaFold",
    "abstract": "Proteins are essential to life.", "authors": "Jumper J",
    "year": "2021", "journal": "Nature",
    "mesh_headings": '["Deep Learning"]', "pub_types": '["Journal Article"]',
    "keywords_author": "[]", "is_open_access": "True", "fulltext_available": "True",
    "abstract_source": "europepmc", "epmc_source": "MED",
    "first_publication_date": "2021-07-15",
}
EVENT = {
    "record_id": STAGED["pid"], "classification": "positive", "rationale": "Develops a method.",
    "model_tier": "flash", "mode": "primary", "prompt_version": "v1",
    "criteria_sha256": "bd9d66dd8929", "batch_id": "classify_flash_staged_20260903",
    "timestamp": "2026-09-03T00:00:00+00:00",
}


def _doc(licensing=None, citations=None):
    return build_document(build_row(STAGED, EVENT, licensing or {}, citations or {}))


def test_the_document_is_the_same_shape_as_every_other_path():
    """This file owns no document shape of its own -- it assembles a row and hands it to the one
    shared builder, so the landscape, curated and incremental paths cannot drift apart."""
    doc = _doc()
    assert set(doc) == {"_id", "schema_version", "identifiers", "publication_metadata",
                        "source", "content_filters", "data_links", "llm_classification",
                        "llm_enrichment"}
    assert doc["schema_version"] == SCHEMA_VERSION


def test_the_pid_becomes_the_document_id():
    """The idempotence property: an incremental re-encounter of the same paper upserts rather than
    duplicating, because the _id is the deterministic UUID5 build_incoming_documents.py minted."""
    assert _doc()["_id"] == STAGED["pid"]


def test_the_classification_comes_from_the_event_not_the_staged_row():
    doc = _doc()
    assert doc["llm_classification"]["classification"] == "positive"
    assert doc["llm_classification"]["batch_id"] == "classify_flash_staged_20260903"
    assert doc["llm_classification"]["provider"] == "deepseek"


def test_incremental_documents_carry_llm_provenance():
    assert _doc()["source"]["decision_provenance"] == "llm"


def test_facets_survive_from_the_staged_row():
    assert _doc()["content_filters"]["mesh_headings"] == ["Deep Learning"]


def test_licence_join_marks_checked_versus_never_looked_up():
    checked = _doc(licensing={"34265844": ("cc by", "Y")})["source"]["access"]
    assert checked["license"] == "cc by" and checked["open_access"] is True
    never = _doc()["source"]["access"]
    assert never["license"] is None  # None means "never looked up", distinct from ""


def test_citations_are_found_by_any_key():
    """lookup_citation tries pmid -> doi -> pmcid, so a count fetched under the doi still lands."""
    doc = _doc(citations={("doi", "10.1038/s41586-021-03819-2"): ("34984", "2026-09-03T00:00:00+00:00")})
    pub = doc["publication_metadata"]
    assert pub["citation_count"] == 34984
    assert pub["citation_source"] == "europepmc"


def test_no_citation_means_null_not_zero():
    pub = _doc()["publication_metadata"]
    assert pub["citation_count"] is None and pub["citation_source"] is None


def test_enrichment_group_is_reserved_and_empty():
    assert all(v is None for v in _doc()["llm_enrichment"].values())


def test_parse_error_events_carry_no_verdict_and_are_ignored(tmp_path):
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EVENT))
        w.writeheader()
        w.writerow({**EVENT, "classification": "parse_error"})
    assert load_classifications(path) == {}


def test_the_latest_usable_verdict_wins(tmp_path):
    """The event log is append-only; a re-run retries a parse error as a new row."""
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EVENT))
        w.writeheader()
        w.writerow({**EVENT, "classification": "negative"})
        w.writerow({**EVENT, "classification": "positive"})
    assert load_classifications(path)[STAGED["pid"]]["classification"] == "positive"


def test_unclassified_staged_records_are_left_out_not_guessed(tmp_path, capsys):
    staged_path, events_path = tmp_path / "staged.csv", tmp_path / "events.csv"
    with staged_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(STAGED))
        w.writeheader()
        w.writerow(STAGED)
        w.writerow({**STAGED, "pid": "unclassified-one"})
    with events_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EVENT))
        w.writeheader()
        w.writerow(EVENT)

    out = tmp_path / "docs.jsonl"
    run(staged_path, events_path, tmp_path / "none.csv", tmp_path / "none2.csv",
        out, tmp_path / "r.json", report_only=False)
    docs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [d["_id"] for d in docs] == [STAGED["pid"]]
    assert "NOT classified" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# v1.3.0 / v1.4.0: the staged row carries the Europe PMC identity and the data-links summary;
# the merged file adds the link detail.
# ---------------------------------------------------------------------------


def test_a_staged_row_without_the_new_columns_still_builds_with_nulls():
    doc = _doc()
    assert doc["identifiers"]["epmc_id"] is None
    assert doc["data_links"]["has_data"] is None and doc["data_links"]["fetched_at"] is None


def test_the_identity_and_summary_ride_in_on_the_staged_row():
    staged = dict(STAGED, epmc_id="34265844", preprint_server="", has_data="Y",
                  data_links_tags='["supporting_data"]', accession_types='["pdb"]',
                  db_cross_references='["PDB"]', has_tm_accessions="Y", has_db_xrefs="Y",
                  has_suppl="Y")
    doc = build_document(build_row(staged, EVENT, {}, {}))
    assert doc["identifiers"]["epmc_id"] == "34265844"
    assert doc["source"]["epmc_source"] == "MED"
    assert doc["publication_metadata"]["preprint_server"] is None
    assert doc["data_links"]["has_data"] is True
    assert doc["data_links"]["accession_types"] == ["pdb"]
    assert doc["data_links"]["fetched_at"] is None     # no merged file: not fetched yet


def test_the_merged_data_links_file_adds_the_detail_by_pid():
    detail = {"fetched_at": "2026-09-14T01:00:00+00:00", "sources": ["epmc_annotations"],
              "link_count": 1, "truncated": False,
              "resources": [{"resource": "pdb", "label": "Protein Data Bank in Europe",
                             "category": "Protein Structures", "id_scheme": "PDBe",
                             "publisher": "Europe PMC", "obtained_by": "tm_accession", "count": 1}],
              "links": [{"resource": "pdb", "id": "6VW1", "url": None, "title": None,
                         "obtained_by": "tm_accession", "relationship": "References",
                         "section": "Article", "frequency": None}]}
    merged = {STAGED["pid"]: {"has_data": "Y", "data_links_tags": '["supporting_data"]',
                              "accession_types": '["pdb"]', "db_cross_references": "[]",
                              "data_links_json": json.dumps(detail)}}
    doc = build_document(build_row(STAGED, EVENT, {}, {}, None, merged))
    assert doc["data_links"]["fetched_at"] == "2026-09-14T01:00:00+00:00"
    assert doc["data_links"]["resources"][0]["resource"] == "pdb"
    assert doc["data_links"]["links"][0]["id"] == "6VW1"
    # a pid the merged file does not cover is untouched
    other = build_document(build_row(dict(STAGED, pid="other"), EVENT, {}, {}, None, merged))
    assert other["data_links"]["fetched_at"] is None


def test_the_identifiers_file_fills_dome_registry_by_pid_and_marks_it_looked_up():
    def dome(identifiers):
        return build_document(build_row(STAGED, EVENT, {}, {}, None, None, identifiers))[
            "identifiers"]["dome_registry"]

    assert dome({STAGED["pid"]: "3mm086r5pw"}) == "3mm086r5pw"
    assert dome({STAGED["pid"]: ""}) == ""
    assert dome({"another-pid": "3mm086r5pw"}) is None
    assert dome(None) is None


def test_load_identifiers_keeps_a_confirmed_miss(tmp_path):
    from build_staged_documents import load_identifiers

    path = tmp_path / "ids.csv"
    path.write_text("pid,dome_registry\np1,3mm086r5pw\np2,\n", encoding="utf-8")
    assert load_identifiers(path) == {"p1": "3mm086r5pw", "p2": ""}
    assert load_identifiers(tmp_path / "missing.csv") == {}
