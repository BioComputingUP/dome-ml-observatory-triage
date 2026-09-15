"""Tests for the v1.6.0 migration: the version plus one constant record_modified, its allowlist,
its inverse and its pre-flight."""

from __future__ import annotations

import json
import re

import pytest

import migrate_v1_6_0 as mig
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


def test_the_migration_sets_the_version_and_one_stamp_within_its_allowlist():
    assert (mig.MODE, mig.FROM_VERSIONS, mig.TO_VERSION) == ("migrate_v1_6_0", ("1.5.1",), "1.6.0")
    stamp = mw.record_modified_stamp()
    assert mig.forward_set(stamp) == {"schema_version": "1.6.0", "record_modified": stamp}
    assert re.match(mw.RECORD_MODIFIED_PATTERN, stamp)
    assert mw.WRITE_MODES["migrate_v1_6_0"] == frozenset({"schema_version", "record_modified"})
    writer = mw.SafeWriter(_FakeMoros(), mode=mig.MODE, run_id="t")
    writer.validate(mig.forward_set(stamp))
    writer.validate(mig.REVERSE_SET)


def test_reverse_restores_the_version_and_removes_the_field_1_5_1_does_not_have():
    assert mig.reverse_update() == {"$set": {"schema_version": "1.5.1"},
                                    "$unset": {"record_modified": ""}}
    assert mig.POPULATED_MARKERS == ()
    mig.reverse_preflight(_FakeMoros())


def test_preflight_refuses_an_unknown_version_and_finishes_a_mixed_state(capsys):
    with pytest.raises(SystemExit, match="refusing to migrate"):
        mig.preflight(_FakeMoros(histogram={"1.5.0": 5}))
    stats = mig.preflight(_FakeMoros(histogram={"1.5.1": 2, "1.6.0": 3}))
    assert (stats["at_from"], stats["at_to"]) == (2, 3)
    assert "mixed state" in capsys.readouterr().out
