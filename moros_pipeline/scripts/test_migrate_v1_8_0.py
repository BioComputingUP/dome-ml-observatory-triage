"""Tests for the v1.8.0 migration: the version plus enrichment_history = [], its allowlist, its inverse
(refused once a superseded enrichment is stored) and its pre-flight."""

from __future__ import annotations

import json

import pytest

import migrate_v1_8_0 as mig
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


def test_the_migration_sets_the_version_and_an_empty_history_within_its_allowlist():
    assert (mig.MODE, mig.FROM_VERSIONS, mig.TO_VERSION) == ("migrate_v1_8_0", ("1.7.0",), "1.8.0")
    assert mig.FORWARD_SET == {"schema_version": "1.8.0", "enrichment_history": []}
    assert mw.WRITE_MODES["migrate_v1_8_0"] == frozenset({"schema_version", "enrichment_history"})
    assert "migrate_v1_8_0" not in mw.STAMPS_RECORD_MODIFIED
    writer = mw.SafeWriter(_FakeMoros(), mode=mig.MODE, run_id="t")
    writer.validate(mig.FORWARD_SET)
    writer.validate(mig.REVERSE_SET)


def test_reverse_restores_the_version_and_removes_the_field():
    assert mig.reverse_update() == {"$set": {"schema_version": "1.7.0"},
                                    "$unset": {"enrichment_history": ""}}
    mig.reverse_preflight(_FakeMoros())


def test_reverse_is_refused_once_history_has_been_written():
    for marker in mig.POPULATED_MARKERS:
        with pytest.raises(SystemExit, match="refusing to reverse"):
            mig.reverse_preflight(_FakeMoros(counts={json.dumps(marker): 1}))


def test_preflight_refuses_1_6_0_and_finishes_a_mixed_state(capsys):
    with pytest.raises(SystemExit, match="migrate_v1_7_0.py first"):
        mig.preflight(_FakeMoros(histogram={"1.6.0": 5}))
    stats = mig.preflight(_FakeMoros(histogram={"1.7.0": 2, "1.8.0": 3}))
    assert (stats["at_from"], stats["at_to"]) == (2, 3)
    assert "mixed state" in capsys.readouterr().out


def test_the_enrichment_mode_can_write_the_history():
    assert "enrichment_history" in mw.WRITE_MODES["enrichment"]
