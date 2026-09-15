"""Tests for resolve_duplicates.py: which copy of a duplicated paper stays, and what stops a delete.

Why these exist: this is the only tool here that deletes documents. It must keep the copy whose `_id`
is what a future fetch will mint (so the corpus stops drifting from Europe PMC), and it must refuse a
group that is not one paper twice -- two Europe PMC records, a curated record, or a copy carrying
something the keeper lacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mongo_landscape_export" / "scripts"))
from pid import mint_landscape_pid

import resolve_duplicates as rd

PMCID, DOI, PMID = "PMC13334259", "10.1186/s12873-026-01647-z", "42310553"


def _doc(pmcid=None, doi=None, pmid=None, doc_id=None, provenance="llm", licence=None, open_access=None,
         citations=None, abstract="an abstract", enrichment=None, dome=None, data_links=None):
    return {
        "_id": doc_id or mint_landscape_pid(pmcid, doi, pmid),
        "identifiers": {"pmcid": pmcid, "doi": doi, "pmid": pmid, "dome_registry": dome},
        "llm_enrichment": {"provider": enrichment},
        "publication_metadata": {"citation_count": citations, "citation_count_updated": "2026-09-01",
                                 "citation_source": "europepmc", "abstract": abstract},
        "source": {"decision_provenance": provenance, "access": {"license": licence, "open_access": open_access}},
        "data_links": {"has_data": data_links},
    }


def _pair(**loser_kwargs):
    """The real shape: an August copy carrying the PMCID, a September copy without it."""
    return [_doc(pmcid=PMCID, doi=DOI, pmid=PMID), _doc(doi=DOI, pmid=PMID, **loser_kwargs)]


def test_the_copy_minted_from_the_combined_identifiers_stays():
    docs = _pair()
    decision = rd.classify_group(docs)
    assert decision["kind"] == "plain"
    assert decision["keeper"] == mint_landscape_pid(PMCID, DOI, PMID) == docs[0]["_id"]
    assert decision["losers"] == [docs[1]["_id"]]


def test_combined_identifiers_take_each_from_whichever_copy_has_it():
    assert rd.combined_identifiers(_pair()) == {"pmcid": PMCID, "doi": DOI, "pmid": PMID}


def test_two_europe_pmc_records_sharing_a_doi_are_left_alone():
    docs = [_doc(pmcid="PMC7144704", doi=DOI, pmid="32308224"),
            _doc(pmcid="PMC7146991", doi=DOI, pmid="31769830")]
    decision = rd.classify_group(docs)
    assert decision["kind"] == "different_pmcids" and decision["losers"] == []


def test_a_curated_or_registry_copy_is_left_alone():
    docs = _pair()
    docs[1]["source"]["decision_provenance"] = "human_curated"
    assert rd.classify_group(docs)["kind"] == "curated_or_registry"


def test_a_group_no_copy_matches_is_left_alone():
    docs = [_doc(doi=DOI, pmid=PMID, doc_id="not-a-minted-id"), _doc(doi=DOI, pmid=PMID, doc_id="nor-this")]
    decision = rd.classify_group(docs)
    assert decision["kind"] == "no_keeper" and decision["losers"] == []


@pytest.mark.parametrize("kwargs, carried", [
    ({"enrichment": "deepseek"}, "an enrichment"),
    ({"dome": "dome-123"}, "a DOME Registry entry"),
    ({"data_links": "Y"}, "data links"),
])
def test_a_copy_carrying_what_no_write_mode_can_move_is_left_for_a_person(kwargs, carried):
    decision = rd.classify_group(_pair(**kwargs))
    assert decision["kind"] == "needs_review" and carried in decision["why"]
    assert decision["losers"] == []


def test_an_abstract_only_on_the_copy_that_would_go_stops_it():
    docs = [_doc(pmcid=PMCID, doi=DOI, pmid=PMID, abstract=""), _doc(doi=DOI, pmid=PMID)]
    assert rd.classify_group(docs)["kind"] == "needs_review"


def test_a_licence_the_keeper_never_looked_up_is_merged_first():
    docs = [_doc(pmcid=PMCID, doi=DOI, pmid=PMID, licence=None),
            _doc(doi=DOI, pmid=PMID, licence="cc by", open_access="Y")]
    decision = rd.classify_group(docs)
    assert decision["kind"] == "merge_then_remove"
    assert decision["merges"]["licences"] == {"source.access.license": "cc by", "source.access.open_access": "Y"}
    assert decision["losers"] == [docs[1]["_id"]]


def test_a_disclosed_licence_beats_the_keepers_looked_up_nothing():
    """"" means the lookup found none on that key; Europe PMC discloses the licence on the PMC record,
    which in these pairs is the other copy."""
    docs = [_doc(pmcid=PMCID, doi=DOI, pmid=PMID, licence=""), _doc(doi=DOI, pmid=PMID, licence="cc by")]
    assert rd.classify_group(docs)["merges"]["licences"]["source.access.license"] == "cc by"


def test_a_licence_the_keeper_already_has_is_left_alone():
    docs = [_doc(pmcid=PMCID, doi=DOI, pmid=PMID, licence="cc by-nc"), _doc(doi=DOI, pmid=PMID, licence="cc by")]
    assert rd.classify_group(docs)["merges"] == {}


def test_a_higher_citation_count_is_merged_with_its_date_and_source():
    docs = [_doc(pmcid=PMCID, doi=DOI, pmid=PMID, citations=3), _doc(doi=DOI, pmid=PMID, citations=11)]
    merges = rd.classify_group(docs)["merges"]
    assert merges["citations"]["publication_metadata.citation_count"] == 11
    assert merges["citations"]["publication_metadata.citation_source"] == "europepmc"


def test_merges_only_name_paths_the_write_modes_allow():
    import moros_write as mw
    merges = rd.classify_group([_doc(pmcid=PMCID, doi=DOI, pmid=PMID, licence=None, citations=1),
                                _doc(doi=DOI, pmid=PMID, licence="cc by", citations=9)])["merges"]
    assert set(merges) == {"licences", "citations"}
    for mode, fields in merges.items():
        assert set(fields) <= set(mw.WRITE_MODES[mode])


def test_a_limit_holds_back_the_rest_for_a_trial():
    groups = {(f"a{i}", f"b{i}"): _pair() for i in range(4)}
    decided, counts = rd.plan_runs(groups, limit=2)
    assert counts["plain"] == 2 and counts["held_back_by_limit"] == 2
    assert sum(len(d["losers"]) for d in decided) == 2


def test_without_a_limit_every_group_is_acted_on():
    groups = {(f"a{i}", f"b{i}"): _pair() for i in range(3)}
    decided, counts = rd.plan_runs(groups, limit=None)
    assert counts["plain"] == 3 and sum(len(d["losers"]) for d in decided) == 3
