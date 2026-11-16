"""Tests for the v1.7.0 migration: the version plus content_filters.model_use = [], its allowlist,
its inverse (refused once e2 content has landed) and its pre-flight."""

from __future__ import annotations

import json

import pytest

import migrate_v1_7_0 as mig
import moros_write as mw


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


def test_the_migration_sets_the_version_and_an_empty_model_use_within_its_allowlist():
    assert (mig.MODE, mig.FROM_VERSIONS, mig.TO_VERSION) == ("migrate_v1_7_0", ("1.6.0",), "1.7.0")
    assert mig.FORWARD_SET == {"schema_version": "1.7.0", "content_filters.model_use": []}
    assert mw.WRITE_MODES["migrate_v1_7_0"] == frozenset({"schema_version", "content_filters.model_use"})
    assert "migrate_v1_7_0" not in mw.STAMPS_RECORD_MODIFIED
    writer = mw.SafeWriter(_FakeMoros(), mode=mig.MODE, run_id="t")
    writer.validate(mig.FORWARD_SET)
    writer.validate(mig.REVERSE_SET)


def test_reverse_restores_the_version_and_removes_the_field():
    assert mig.reverse_update() == {"$set": {"schema_version": "1.6.0"},
                                    "$unset": {"content_filters.model_use": ""}}
    mig.reverse_preflight(_FakeMoros())


def test_reverse_is_refused_once_e2_content_has_landed():
    for marker in mig.POPULATED_MARKERS:
        with pytest.raises(SystemExit, match="refusing to reverse"):
            mig.reverse_preflight(_FakeMoros(counts={json.dumps(marker): 1}))


def test_preflight_refuses_an_unknown_version_and_finishes_a_mixed_state(capsys):
    with pytest.raises(SystemExit, match="refusing to migrate"):
        mig.preflight(_FakeMoros(histogram={"1.5.1": 5}))
    stats = mig.preflight(_FakeMoros(histogram={"1.6.0": 2, "1.7.0": 3}))
    assert (stats["at_from"], stats["at_to"]) == (2, 3)
    assert "mixed state" in capsys.readouterr().out


def test_the_enrichment_mode_can_write_model_use():
    assert "content_filters.model_use" in mw.WRITE_MODES["enrichment"]
