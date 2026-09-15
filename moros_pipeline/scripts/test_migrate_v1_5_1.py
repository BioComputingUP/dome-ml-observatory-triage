"""Tests for the v1.5.1 migration: the version stamp only, its allowlist, and its pre-flight."""

from __future__ import annotations

import json

import pytest

import migrate_v1_5_1 as mig
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


def test_the_migration_sets_only_the_version_and_its_allowlist_is_only_the_version():
    assert (mig.MODE, mig.FROM_VERSIONS, mig.TO_VERSION) == ("migrate_v1_5_1", ("1.5.0",), "1.5.1")
    assert mig.FORWARD_SET == {"schema_version": "1.5.1"}
    assert mig.REVERSE_SET == {"schema_version": "1.5.0"}
    assert mw.WRITE_MODES["migrate_v1_5_1"] == frozenset({"schema_version"})


def test_reverse_is_always_safe_because_nothing_lands_in_documents():
    assert mig.POPULATED_MARKERS == ()
    mig.reverse_preflight(_FakeMoros())


def test_preflight_refuses_an_unknown_version_and_finishes_a_mixed_state(capsys):
    with pytest.raises(SystemExit, match="refusing to migrate"):
        mig.preflight(_FakeMoros(histogram={"1.4.0": 5}))
    stats = mig.preflight(_FakeMoros(histogram={"1.5.0": 2, "1.5.1": 3}))
    assert (stats["at_from"], stats["at_to"]) == (2, 3)
    assert "mixed state" in capsys.readouterr().out
