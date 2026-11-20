"""Tests for the ingestion-side vocabulary cleaning (rule set c1), on the real release vocabularies."""

from __future__ import annotations

from pathlib import Path

import pytest

import enrichment_clean as ec

VOCAB = ec.load_vocab()


def _clean(**values):
    return ec.clean_values(values, VOCAB)


def test_the_vocabularies_are_the_current_release_ones_with_their_caps():
    assert ec.MODELLING_VOCAB.parent.name == "e3"
    assert VOCAB.caps == {"domain_tier1": 1, "domain_tier2": 2, "domain_tier3": 3,
                          "learning_paradigm": 2, "model_family": 3, "model_use": 2}


def test_a_clean_record_is_left_exactly_as_it_is():
    values = {"domain_tier1": "Omics", "domain_tier2": ["Genomics"], "domain_tier3": [],
              "learning_paradigm": ["supervised"], "model_family": ["deep learning"],
              "model_use": ["trained in this study"]}
    cleaned, changes = ec.clean_values(values, VOCAB)
    assert (cleaned, changes) == (values, [])


def test_case_is_normalised_to_the_label():
    cleaned, changes = _clean(learning_paradigm=["Supervised"])
    assert cleaned["learning_paradigm"] == ["supervised"]
    assert changes == ["cleaned:c1:learning_paradigm:case:Supervised->supervised"]


def test_a_tier2_term_in_tier3_moves_up_when_tier2_has_room():
    cleaned, changes = _clean(domain_tier2=["Genomics"], domain_tier3=["Sequence analysis"])
    assert cleaned == {"domain_tier2": ["Genomics", "Sequence analysis"], "domain_tier3": []}
    assert changes == ["cleaned:c1:domain_tier3:moved_to_domain_tier2:Sequence analysis"]


def test_a_tier2_term_in_tier3_is_dropped_when_tier2_is_full():
    cleaned, changes = _clean(domain_tier2=["Genomics", "Proteomics"], domain_tier3=["Sequence analysis"])
    assert cleaned["domain_tier2"] == ["Genomics", "Proteomics"] and cleaned["domain_tier3"] == []
    assert changes == ["cleaned:c1:domain_tier3:dropped_full:Sequence analysis"]


def test_a_paradigm_named_as_a_family_is_moved_or_recognised_as_a_duplicate():
    cleaned, changes = _clean(learning_paradigm=["supervised"], model_family=["reinforcement"])
    assert cleaned == {"learning_paradigm": ["supervised", "reinforcement"], "model_family": []}
    assert changes == ["cleaned:c1:model_family:moved_to_learning_paradigm:reinforcement"]
    cleaned, changes = _clean(learning_paradigm=["reinforcement"], model_family=["reinforcement"])
    assert cleaned["model_family"] == [] and changes == [
        "cleaned:c1:model_family:dropped_duplicate:reinforcement"]


def test_a_term_of_no_field_is_dropped():
    cleaned, changes = _clean(domain_tier2=["Geology"], domain_tier1="Cheminformics")
    assert cleaned == {"domain_tier2": [], "domain_tier1": None}
    assert set(changes) == {"cleaned:c1:domain_tier2:dropped_unknown:Geology",
                            "cleaned:c1:domain_tier1:dropped_unknown:Cheminformics"}


def test_over_the_cap_keeps_the_first_n_in_the_models_order():
    tier3 = ["Biobank", "Biodiversity", "Bioengineering", "Bioimaging"]
    cleaned, changes = _clean(domain_tier3=tier3)
    assert cleaned["domain_tier3"] == tier3[:3]
    assert changes == ["cleaned:c1:domain_tier3:trimmed:Bioimaging"]


def test_repeats_are_kept_once():
    cleaned, changes = _clean(model_family=["deep learning", "deep learning"])
    assert cleaned["model_family"] == ["deep learning"]
    assert changes == ["cleaned:c1:model_family:dedup:deep learning"]


def test_a_move_never_creates_a_field_the_record_does_not_carry():
    # An e1 record has no model_use: a model_use term elsewhere is dropped, not written into it.
    cleaned, changes = _clean(model_family=["trained in this study"])
    assert cleaned == {"model_family": []} and "model_use" not in cleaned
    assert changes == ["cleaned:c1:model_family:dropped_unknown:trained in this study"]


def test_cleaning_is_idempotent():
    values = {"domain_tier1": "omics", "domain_tier2": ["Genomics", "Geology"],
              "domain_tier3": ["Sequence analysis", "Proteins", "Proteins"],
              "learning_paradigm": [], "model_family": ["reinforcement", "deep learning"]}
    once, _ = ec.clean_values(values, VOCAB)
    twice, changes = ec.clean_values(once, VOCAB)
    assert twice == once and changes == []


def test_clean_fields_works_on_load_updates_and_appends_to_vocab_violations():
    fields = {"content_filters.domain_tier3": ["Sequence analysis"],
              "content_filters.domain_tier2": [],
              "content_filters.model_type": ["Geology"],
              "llm_enrichment.vocab_violations": ["domain_tier3:unknown:Sequence analysis"]}
    out, changes = ec.clean_fields(fields, VOCAB)
    assert out["content_filters.domain_tier2"] == ["Sequence analysis"]
    assert out["content_filters.domain_tier3"] == []
    assert out["content_filters.model_type"] == ["Geology"]          # open vocabulary: untouched
    assert out["llm_enrichment.vocab_violations"] == [
        "domain_tier3:unknown:Sequence analysis",
        "cleaned:c1:domain_tier3:moved_to_domain_tier2:Sequence analysis"]
    assert fields["content_filters.domain_tier3"] == ["Sequence analysis"]   # input not mutated


def _runs_events() -> list[Path]:
    parent = Path(__file__).resolve().parents[3]
    for name in ("dome-ml-observatory-enrichment-runs", "dome-observatory-enrichment-runs"):
        found = sorted((parent / name / "events").glob("*/*_events.csv"))
        if found:
            return found
    return []


@pytest.mark.skipif(not _runs_events(), reason="the private runs repository is not checked out beside this one")
def test_the_first_e3_production_load_cleans_the_records_measured_on_2026_11_20():
    paths = [p for p in _runs_events() if p.parent.name in ("core_bioinformatics", "batch2_high_impact")]
    if len(paths) != 21:
        pytest.skip("the runs repository holds other cohorts or is partial")
    result = ec.audit(paths, VOCAB)
    assert (result["records"], result["records_cleaned"]) == (100_073, 277)
