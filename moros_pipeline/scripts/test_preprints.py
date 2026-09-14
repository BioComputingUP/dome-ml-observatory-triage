"""Tests for the v1.3.0 preprint / Europe PMC identity backfill: the field load and its allowlist.

The fetch side (`fetch_epmc_metadata.py`, PPR-first selection for preprints) is tested in
`test_epmc_metadata.py`. Here: `load_fields.py --mode preprints` and `WRITE_MODES["preprints"]`.
"""

from __future__ import annotations

import load_fields as lf
import moros_write as mw


def test_a_preprint_row_writes_all_three_fields():
    pid, update = lf.preprint_row_to_update(
        {"pid": "p1", "epmc_id": "PPR18364", "epmc_source": "PPR", "preprint_server": "bioRxiv"})
    assert pid == "p1"
    assert update == {
        "identifiers.epmc_id": "PPR18364",
        "source.epmc_source": "PPR",
        "publication_metadata.preprint_server": "bioRxiv",
    }


def test_a_medline_row_writes_identity_but_no_server_key():
    """A MED record has no publisher; writing an explicit null would be a no-op dressed up as an
    answer, and re-running the load must change zero documents."""
    _, update = lf.preprint_row_to_update(
        {"pid": "p1", "epmc_id": "41466298", "epmc_source": "MED", "preprint_server": ""})
    assert update == {"identifiers.epmc_id": "41466298", "source.epmc_source": "MED"}
    assert "publication_metadata.preprint_server" not in update


def test_a_half_identity_yields_no_update():
    assert lf.preprint_row_to_update({"pid": "p1", "epmc_id": "", "epmc_source": "MED"}) is None
    assert lf.preprint_row_to_update({"pid": "p1", "epmc_id": "1", "epmc_source": ""}) is None
    assert lf.preprint_row_to_update({"pid": "", "epmc_id": "1", "epmc_source": "MED"}) is None


def test_the_preprints_allowlist_is_exactly_three_fields_plus_the_version():
    assert mw.WRITE_MODES["preprints"] == frozenset({
        "schema_version", "identifiers.epmc_id", "source.epmc_source",
        "publication_metadata.preprint_server",
    })


def test_the_preprints_mode_cannot_reach_what_decides_a_record():
    allowed = mw.WRITE_MODES["preprints"]
    assert not any(p.startswith("llm_classification") for p in allowed)
    assert not any(p.startswith("llm_enrichment") for p in allowed)
    assert "source.decision_provenance" not in allowed
    assert "publication_metadata.journal" not in allowed
    assert "content_filters.pub_types" not in allowed


def test_every_mapper_output_path_is_inside_its_allowlist():
    _, update = lf.preprint_row_to_update(
        {"pid": "p", "epmc_id": "PPR1", "epmc_source": "PPR", "preprint_server": "medRxiv"})
    assert set(update) <= mw.WRITE_MODES["preprints"]
