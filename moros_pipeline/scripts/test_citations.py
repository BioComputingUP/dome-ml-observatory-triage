"""Tests for the citation fetch keying and the shared citation index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                       / "mongo_landscape_export" / "scripts"))

from citations_index import load_citation_index, lookup_citation  # noqa: E402

from fetch_citations import assign_key, build_query, pick_best  # noqa: E402


# -- key assignment ---------------------------------------------------------


def test_pmid_wins_when_present():
    assert assign_key("34265844", "PMC8371605", "10.1/x") == ("pmid", "34265844")


def test_doi_is_used_when_there_is_no_pmid():
    # 67,985 records in the corpus have no pmid; a pmid-only fetch would leave 8.2% null forever.
    assert assign_key("", "PMC1", "10.1038/ABC") == ("doi", "10.1038/abc")


def test_pmcid_is_the_last_resort():
    assert assign_key("", "PMC1", "") == ("pmcid", "PMC1")


def test_a_record_with_no_identifier_is_not_fetched():
    assert assign_key("", "", "") is None


def test_dois_are_lowercased_for_a_stable_join_key():
    assert assign_key("", "", "10.1038/AbC")[1] == "10.1038/abc"


# -- query construction -----------------------------------------------------


def test_pmid_clauses_are_unquoted_and_restricted_to_med():
    """A quoted single-clause chunk combined with AND SRC:MED silently returns 0 hits on EPMC's
    Lucene parser -- a real prior incident, documented in ingest/epmc_client.py."""
    query = build_query("pmid", ["34265844", "38718835"])
    assert query == "(EXT_ID:34265844 OR EXT_ID:38718835) AND SRC:MED"
    assert '"' not in query


def test_doi_clauses_are_quoted_and_not_source_restricted():
    """DOIs contain / and . and must be quoted. SRC:MED must NOT be applied: the doi pass exists
    precisely to reach the preprint and PMC-only records a MED restriction would exclude."""
    query = build_query("doi", ["10.1038/s41586-021-03819-2"])
    assert query == '(DOI:"10.1038/s41586-021-03819-2")'
    assert "SRC:MED" not in query


def test_pmcid_clauses_are_not_source_restricted():
    assert build_query("pmcid", ["PMC8371605"]) == "(PMCID:PMC8371605)"


# -- multi-source resolution ------------------------------------------------


def test_med_beats_a_preprint_for_the_same_doi():
    """A 50-DOI query really did return 51 records: one DOI existed as both a MED article and a
    PPR preprint, which carry different counts."""
    best = pick_best([
        {"source": "PPR", "citedByCount": 3, "id": "PPR1"},
        {"source": "MED", "citedByCount": 41, "id": "MED1"},
    ])
    assert best["id"] == "MED1"


def test_resolution_does_not_depend_on_response_order():
    records = [
        {"source": "PMC", "citedByCount": 7, "id": "PMC1"},
        {"source": "MED", "citedByCount": 2, "id": "MED1"},
    ]
    assert pick_best(records)["id"] == pick_best(list(reversed(records)))["id"] == "MED1"


def test_higher_count_breaks_a_tie_within_one_source():
    best = pick_best([
        {"source": "MED", "citedByCount": 5, "id": "a"},
        {"source": "MED", "citedByCount": 50, "id": "b"},
    ])
    assert best["id"] == "b"


# -- the shared index -------------------------------------------------------


def _index(tmp_path, rows: list[str]) -> Path:
    path = tmp_path / "citations.csv"
    path.write_text(
        "key_type,key,epmc_source,pmid,pmcid,doi,citation_count,citation_source,fetched_at\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_lookup_finds_a_count_under_any_of_the_three_keys(tmp_path):
    path = _index(tmp_path, ["doi,10.1/abc,MED,,,10.1/abc,42,europepmc,2026-09-03T00:00:00+00:00"])
    index = load_citation_index(path)
    # Fetched under doi, but the record also has a pmid -- the join must still find it.
    assert lookup_citation(index, pmid="999", doi="10.1/ABC")[:2] == ("42", "2026-09-03T00:00:00+00:00")


def test_lookup_returns_which_key_matched(tmp_path):
    path = _index(tmp_path, ["pmcid,PMC1,PMC,,PMC1,,7,europepmc,t"])
    assert lookup_citation(load_citation_index(path), pmcid="PMC1")[2] == "pmcid"


def test_a_record_with_no_count_anywhere_returns_none(tmp_path):
    path = _index(tmp_path, ["pmid,1,MED,1,,,5,europepmc,t"])
    assert lookup_citation(load_citation_index(path), pmid="2", doi="10.1/z") is None


def test_rows_with_no_count_are_skipped_not_stored_as_empty(tmp_path):
    """'EPMC returned the record but no count' and 'the count is zero' must not collapse."""
    path = _index(tmp_path, ["pmid,1,MED,1,,,,europepmc,t", "pmid,2,MED,2,,,0,europepmc,t"])
    index = load_citation_index(path)
    assert ("pmid", "1") not in index
    assert index[("pmid", "2")][0] == "0"


def test_a_later_row_supersedes_an_earlier_one_for_the_same_key(tmp_path):
    """A refresh appends to the same file; the newer count must win."""
    path = _index(tmp_path, ["pmid,1,MED,1,,,5,europepmc,2026-01-01",
                             "pmid,1,MED,1,,,9,europepmc,2026-09-03"])
    assert load_citation_index(path)[("pmid", "1")] == ("9", "2026-09-03")
