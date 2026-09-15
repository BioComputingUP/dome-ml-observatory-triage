"""Tests for the v1.5.0 identifiers load: the row mapper, the `identifiers` allowlist, and the
version-only migration that precedes it."""

from __future__ import annotations

import json

import pytest

import load_fields as lf
import migrate_v1_5_0 as mig
import moros_write as mw


def test_a_found_entry_and_a_confirmed_miss_are_both_written():
    assert lf.identifiers_row_to_update({"pid": "p1", "dome_registry": "3mm086r5pw"}) == (
        "p1", {"identifiers.dome_registry": "3mm086r5pw"})
    assert lf.identifiers_row_to_update({"pid": "p2", "dome_registry": ""}) == (
        "p2", {"identifiers.dome_registry": ""})


def test_a_row_without_identifier_columns_or_without_a_pid_yields_nothing():
    assert lf.identifiers_row_to_update({"pid": "p1"}) is None
    assert lf.identifiers_row_to_update({"pid": "", "dome_registry": "3mm086r5pw"}) is None


def test_an_unclean_identifier_cannot_reach_the_writer():
    with pytest.raises(ValueError, match="not a clean identifier"):
        lf.identifiers_row_to_update({"pid": "p1", "dome_registry": "3mm 086r5pw"})


def test_the_identifiers_allowlist_is_the_five_reserved_fields_plus_the_version():
    assert mw.WRITE_MODES["identifiers"] == frozenset({
        "schema_version", "identifiers.dome_registry", "identifiers.bioai_repo",
        "identifiers.huggingface", "identifiers.kaggle", "identifiers.zenodo",
    })


def test_the_identifiers_mode_cannot_reach_links_the_epmc_identity_or_verdicts():
    allowed = mw.WRITE_MODES["identifiers"]
    assert {"identifiers.epmc_id", "identifiers.pmid", "identifiers.pmcid", "identifiers.doi"}.isdisjoint(
        allowed)
    assert not any(p.startswith(("data_links", "llm_classification", "llm_enrichment", "source",
                                 "publication_metadata", "content_filters")) for p in allowed)
    assert not any(p.startswith("identifiers") for p in mw.WRITE_MODES["data_links"])


def test_every_identifiers_mapper_path_is_inside_its_allowlist_and_the_mode_is_wired():
    _, update = lf.identifiers_row_to_update({"pid": "p", "dome_registry": "x1",
                                              "zenodo": "10.5281/zenodo.1"})
    assert set(update) <= mw.WRITE_MODES["identifiers"]
    assert lf.DEFAULT_INPUTS["identifiers"].name == "pid_identifiers.csv"
    assert "identifiers" in lf.ROW_MAPPERS


def test_a_dirty_identifiers_file_is_refused_before_moros_is_contacted(tmp_path, monkeypatch):
    path = tmp_path / "pid_identifiers.csv"
    path.write_text("pid,dome_registry\np0,3mm086r5pw\np1,bad id\n", encoding="utf-8")

    def no_connect(*args, **kwargs):
        raise AssertionError("moros must not be contacted")

    monkeypatch.setattr(lf.Moros, "from_env", no_connect)
    with pytest.raises(SystemExit, match="not a clean identifier"):
        lf.run("identifiers", path, confirm=True, limit=None)


# -- migrate_v1_5_0 ---------------------------------------------------------------------------


class _FakeMoros:
    def __init__(self, histogram=None, counts=None):
        self._histogram = histogram or {}
        self._counts = counts or {}

    def histogram(self, field):
        return self._histogram

    def count(self, query):
        return self._counts.get(json.dumps(query), 0)

    def describe(self):
        return "fake"


def test_the_migration_sets_only_the_version_and_its_allowlist_is_only_the_version():
    assert mig.FORWARD_SET == {"schema_version": "1.5.0"}
    assert mig.REVERSE_SET == {"schema_version": "1.4.0"}
    assert mw.WRITE_MODES["migrate_v1_5_0"] == frozenset({"schema_version"})


def test_reverse_is_refused_once_v1_5_0_content_has_landed():
    for marker in mig.POPULATED_MARKERS:
        with pytest.raises(SystemExit, match="refusing to reverse"):
            mig.reverse_preflight(_FakeMoros(counts={json.dumps(marker): 1}))
    mig.reverse_preflight(_FakeMoros())


def test_preflight_refuses_an_unknown_version_and_finishes_a_mixed_state(capsys):
    with pytest.raises(SystemExit, match="refusing to migrate"):
        mig.preflight(_FakeMoros(histogram={"1.2.0": 5}))
    stats = mig.preflight(_FakeMoros(histogram={"1.4.0": 2, "1.5.0": 3}))
    assert (stats["at_from"], stats["at_to"]) == (2, 3)
    assert "mixed state" in capsys.readouterr().out
