"""Tests for the hierarchical domain rendering and the wrong-tier repair.

Both exist to reduce this step's output-token cost without losing tagging accuracy: the tree gives
the model back the structure a flat 259-term list throws away, and the repair recovers the
violation class the thinking-OFF ablation found was 46% of all violations -- a real vocabulary term
filed under the wrong tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dome_triage.llm_classify import enrichment as en

CRITERIA = Path(__file__).resolve().parent.parent / "curation_criteria"


@pytest.fixture(scope="module")
def vocabs() -> dict:
    return en.load_vocabularies(CRITERIA)


@pytest.fixture(scope="module")
def lookup(vocabs) -> dict:
    return en._build_lookup(vocabs)


def _all_labels(vocabs) -> list[str]:
    return [t["label"] for f in vocabs["domain"]["fields"].values() for t in f["terms"]]


# ---------------------------------------------------------------------------
# The tree rendering
# ---------------------------------------------------------------------------


def test_domain_tree_renders_every_term_exactly_once(vocabs):
    """The failure that would matter: a term silently absent from the prompt can never be chosen,
    so the vocabulary would quietly shrink."""
    tree = en.build_static_system_text(vocabs, "tree")
    labels = _all_labels(vocabs)
    assert len(labels) == 259
    for label in labels:
        assert tree.count(f"{label}\n") + tree.count(f"{label} ") >= 1, f"{label!r} missing"


def test_the_tree_names_all_three_tiers_and_their_caps(vocabs):
    tree = en.build_static_system_text(vocabs, "tree")
    for tier in ("domain_tier1", "domain_tier2", "domain_tier3"):
        assert tier in tree
    assert "never move a term into a different tier" in tree


def test_flat_rendering_is_unchanged_so_existing_hashes_still_resume(vocabs):
    """Adding parent_ids to domain_vocab.json must not disturb the flat text -- the completed
    3,611-record trial and the Bioinformatics run both resume on vocab_sha256 41db952f1511."""
    flat = en.build_static_system_text(vocabs, "flat")
    assert en.vocab_sha256(flat).startswith("41db952f1511")


def test_the_two_renderings_hash_differently(vocabs):
    """A rendering change must start a fresh batch rather than conflating two prompt variants in
    one event log."""
    flat = en.vocab_sha256(en.build_static_system_text(vocabs, "flat"))
    tree = en.vocab_sha256(en.build_static_system_text(vocabs, "tree"))
    assert flat != tree


def test_flat_is_the_default(vocabs):
    assert en.build_static_system_text(vocabs) == en.build_static_system_text(vocabs, "flat")


def test_an_unknown_rendering_is_refused(vocabs):
    with pytest.raises(ValueError, match="domain_rendering must be one of"):
        en.build_static_system_text(vocabs, "nested")


def test_the_tree_is_byte_stable_across_calls(vocabs):
    """It is the prefix-cache boundary; EDAM is a DAG and 51 terms have several parents, so the
    parent chosen must not vary run to run."""
    assert en.build_static_system_text(vocabs, "tree") == en.build_static_system_text(vocabs, "tree")


# ---------------------------------------------------------------------------
# Wrong-tier repair
# ---------------------------------------------------------------------------


def _parse(lookup, **fields) -> dict:
    payload = {f: [] for f in en.LIST_FIELDS}
    payload["rationale"] = "r"
    payload.update(fields)
    return en.parse_enrichment(json.dumps(payload), lookup)


def test_a_tier2_term_filed_under_tier3_is_moved_to_its_real_tier(lookup):
    """The exact shape of 41 of the ablation's violations."""
    result = _parse(lookup, domain_tier3=["Genomics"])
    assert result["domain_tier2"] == ["Genomics"]
    assert result["domain_tier3"] == []
    assert result["vocab_violations"] == ["domain_tier3:repaired_to_domain_tier2:Genomics"]


def test_a_repair_is_recorded_distinctly_from_a_genuine_miss(lookup):
    """Repair must never be able to flatter the quality metric -- the two are counted apart."""
    result = _parse(lookup, domain_tier3=["Genomics", "Kombucha brewing"])
    violations = result["vocab_violations"]
    assert any(v.startswith("domain_tier3:repaired_to_") for v in violations)
    assert "domain_tier3:unknown:Kombucha brewing" in violations


def test_a_genuinely_unknown_term_is_still_kept_never_coerced(lookup):
    result = _parse(lookup, domain_tier2=["Kombucha brewing"])
    assert result["domain_tier2"] == ["Kombucha brewing"]
    assert result["vocab_violations"] == ["domain_tier2:unknown:Kombucha brewing"]


def test_repair_respects_the_destination_cap(lookup):
    """domain_tier1 holds at most 1. A second tier-1 term arriving from another tier cannot
    silently exceed that, and must not be dropped either."""
    result = _parse(lookup, domain_tier1=["Biology"], domain_tier2=["Medicine"])
    assert result["domain_tier1"] == ["Biology"]
    assert result["domain_tier2"] == ["Medicine"]  # left where it was, flagged
    assert "domain_tier2:unknown:Medicine" in result["vocab_violations"]


def test_repair_does_not_duplicate_a_term_the_destination_already_has(lookup):
    result = _parse(lookup, domain_tier2=["Genomics"], domain_tier3=["Genomics"])
    assert result["domain_tier2"] == ["Genomics"]
    assert result["domain_tier3"] == ["Genomics"]  # kept, flagged, not duplicated upward
    assert result["domain_tier2"].count("Genomics") == 1


def test_correctly_placed_terms_produce_no_violation(lookup):
    result = _parse(lookup, domain_tier1=["Biology"], domain_tier2=["Genomics"],
                    learning_paradigm=["supervised"])
    assert result["vocab_violations"] == []
    assert result["parse_status"] == "ok"


def test_the_real_smoke_run_violation_is_repaired(lookup):
    """`domain_tier3:unknown:Proteins` was one of the two real violations in the 38-record
    Bioinformatics run. Proteins is a genuine tier-2 term, so it is a misfiling, not a miss."""
    result = _parse(lookup, domain_tier3=["Proteins"])
    assert result["domain_tier2"] == ["Proteins"]
    assert result["vocab_violations"] == ["domain_tier3:repaired_to_domain_tier2:Proteins"]


def test_closed_modelling_fields_are_untouched_by_repair(lookup):
    """learning_paradigm and model_family are separate vocabularies, not tiers of one tree --
    a term must never migrate between them."""
    result = _parse(lookup, learning_paradigm=["deep learning"])
    assert result["learning_paradigm"] == ["deep learning"]
    assert result["model_family"] == []
    assert "learning_paradigm:unknown:deep learning" in result["vocab_violations"]


def test_cap_exceeded_is_still_reported(lookup):
    result = _parse(lookup, domain_tier2=["Genomics", "Proteins", "Oncology"])
    assert any("cap_exceeded" in v for v in result["vocab_violations"])
