"""Tests for the search-space definition and the coverage ledger."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from coverage_ledger import CoverageLedger, SearchSpace, missing_years

BASE = {
    "name": "test_space",
    "terms": ['"artificial intelligence"', '"machine learning"'],
    "combine": "OR",
    "sources": [],
    "date_field": "FIRST_PDATE",
    "coverage_start": "2020-01-01",
}


def _space(tmp_path, **overrides) -> SearchSpace:
    path = tmp_path / "search_space.yaml"
    path.write_text(yaml.safe_dump({**BASE, **overrides}), encoding="utf-8")
    return SearchSpace.load(path)


def test_query_shape_matches_the_pipelines_own(tmp_path):
    # Must be byte-identical in shape to ingest/bulk_match.py::_range_query, or windows fetched by
    # the two paths are not comparable.
    space = _space(tmp_path)
    assert space.query("2021-01-01", "2021-12-31") == (
        '("artificial intelligence" OR "machine learning") '
        "AND (FIRST_PDATE:[2021-01-01 TO 2021-12-31])"
    )


def test_sources_are_only_added_when_configured(tmp_path):
    assert "SRC:" not in _space(tmp_path).query("2021-01-01", "2021-12-31")
    restricted = _space(tmp_path, sources=["MED", "PPR"]).query("2021-01-01", "2021-12-31")
    assert restricted.endswith("AND (SRC:MED OR SRC:PPR)")


def test_editing_the_terms_invalidates_coverage(tmp_path):
    before = _space(tmp_path)
    after = _space(tmp_path, terms=BASE["terms"] + ['"deep learning"'])
    assert before.sha256() != after.sha256()


def test_reordering_terms_does_not_invalidate_coverage(tmp_path):
    # A cosmetic YAML edit must not throw away real fetch history.
    before = _space(tmp_path)
    after = _space(tmp_path, terms=list(reversed(BASE["terms"])))
    assert before.sha256() == after.sha256()


def test_changing_the_date_field_invalidates_coverage(tmp_path):
    assert _space(tmp_path).sha256() != _space(tmp_path, date_field="PUB_YEAR").sha256()


def test_an_empty_term_list_is_refused(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump({**BASE, "terms": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no terms"):
        SearchSpace.load(path)


def test_ledger_records_and_reloads_windows(tmp_path):
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    ledger.record_window(space, "2021-01-01", "2021-12-31", fetched=1234, path="a.jsonl")
    ledger.record_loaded(space, "2021-01-01", "2021-12-31", loaded=1200)
    ledger.save()

    reloaded = CoverageLedger(tmp_path / "ledger.json")
    window = reloaded.entry(space)["windows"][0]
    assert window["fetched"] == 1234 and window["loaded_to_moros"] == 1200
    assert reloaded.covered_years(space) == {2021}


def test_a_different_query_sees_none_of_the_first_querys_coverage(tmp_path):
    original = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    ledger.record_window(original, "2021-01-01", "2021-12-31", fetched=10, path="a")
    widened = _space(tmp_path, terms=BASE["terms"] + ['"deep learning"'])
    assert ledger.covered_years(original) == {2021}
    assert ledger.covered_years(widened) == set()


def test_missing_years_spans_coverage_start_to_today(tmp_path):
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    assert missing_years(space, ledger, date(2023, 6, 1)) == [2020, 2021, 2022, 2023]


def test_the_current_year_is_always_refetched(tmp_path):
    """It is still filling up, so 'already fetched' is never true for it."""
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    for year in (2020, 2021, 2022, 2023):
        ledger.record_window(space, f"{year}-01-01", f"{year}-12-31", fetched=1, path="x")
    assert missing_years(space, ledger, date(2023, 6, 1)) == [2023]


def test_recording_a_window_twice_updates_rather_than_duplicates(tmp_path):
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    ledger.record_window(space, "2021-01-01", "2021-12-31", fetched=1, path="a")
    ledger.record_window(space, "2021-01-01", "2021-12-31", fetched=99, path="b")
    windows = ledger.entry(space)["windows"]
    assert len(windows) == 1 and windows[0]["fetched"] == 99


def test_marking_an_unfetched_window_as_loaded_is_an_error(tmp_path):
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    with pytest.raises(KeyError):
        ledger.record_loaded(space, "2021-01-01", "2021-12-31", loaded=5)


def test_a_year_fetched_only_partway_is_fetched_again_once_it_has_ended(tmp_path):
    """A refresh with --up-to before 31 December leaves a partial window. When the calendar turns,
    the rest of that year must still be fetched rather than skipped as covered."""
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    for year in (2020, 2021, 2022):
        ledger.record_window(space, f"{year}-01-01", f"{year}-12-31", fetched=1, path="x")
    ledger.record_window(space, "2023-01-01", "2023-12-15", fetched=1, path="x")
    assert missing_years(space, ledger, date(2024, 1, 20)) == [2023, 2024]


def test_a_partial_year_still_counts_for_the_cross_check(tmp_path):
    """The cross-check against moros asks whether a year was fetched at all, so a partial window must
    not make moros look ahead of the ledger."""
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    ledger.record_window(space, "2023-01-01", "2023-09-03", fetched=1, path="x")
    assert ledger.covered_years(space) == {2023}
    assert ledger.complete_years(space) == set()
