"""Tests for the search-space definition and the coverage ledger."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from coverage_ledger import CoverageLedger, SearchSpace, missing_years, plan_index_window

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


def test_index_query_searches_first_index_date_without_a_publication_bound(tmp_path):
    space = _space(tmp_path)
    assert space.index_query("2026-09-03", "2026-09-10") == (
        '("artificial intelligence" OR "machine learning") '
        "AND (FIRST_IDATE:[2026-09-03 TO 2026-09-10])"
    )


def test_the_first_index_window_also_takes_papers_dated_after_the_last_year_window(tmp_path):
    q = _space(tmp_path, sources=["MED"]).index_query("2026-09-03", "2026-09-10", "2026-09-04")
    assert q == ('("artificial intelligence" OR "machine learning") AND (FIRST_IDATE:[2026-09-03 TO '
                 '2026-09-10] OR FIRST_PDATE:[2026-09-04 TO 2100-12-31]) AND (SRC:MED)')


def test_index_windows_do_not_change_the_query_hash(tmp_path):
    space = _space(tmp_path)
    ledger = CoverageLedger(tmp_path / "ledger.json")
    digest = space.sha256()
    ledger.record_index_window(space, "2026-09-03", "2026-09-10", fetched=5, path="x")
    assert ledger.entry(space)["query_sha256"] == digest


def _ledger_with_year_window(tmp_path, to="2026-09-03", fetched_on="2026-09-03"):
    space = _space(tmp_path, coverage_start="2025-01-01")
    ledger = CoverageLedger(tmp_path / "ledger.json")
    ledger.record_window(space, "2025-01-01", "2025-12-31", fetched=1, path="x")
    ledger.record_window(space, "2026-01-01", to, fetched=1, path="x")
    for w in ledger.entry(space)["windows"]:
        w["fetched_at"] = f"{fetched_on}T20:00:00+00:00"
    return space, ledger


def test_indexed_through_is_the_day_the_year_window_ran(tmp_path):
    space, ledger = _ledger_with_year_window(tmp_path, to="2026-09-01", fetched_on="2026-09-03")
    assert ledger.indexed_through(space) == "2026-09-03"
    assert ledger.future_dated_from(space) == "2026-09-02"


def test_an_index_window_covers_only_to_its_end_even_when_fetched_later(tmp_path):
    """A run bounded with --up-to must not let the next run start after the days it did not reach."""
    space, ledger = _ledger_with_year_window(tmp_path)
    ledger.record_index_window(space, "2026-09-03", "2026-09-10", fetched=5, path="x")
    ledger.entry(space)["index_windows"][0]["fetched_at"] = "2026-09-15T21:00:00+00:00"
    assert ledger.indexed_through(space) == "2026-09-10"
    assert ledger.future_dated_from(space) is None


def test_plan_last_starts_where_coverage_ends(tmp_path):
    space, ledger = _ledger_with_year_window(tmp_path)
    assert plan_index_window(space, ledger, date(2026, 9, 10), "last") == ("2026-09-03", "2026-09-04")


def test_plan_refuses_a_gap_but_accepts_an_overlap(tmp_path):
    space, ledger = _ledger_with_year_window(tmp_path)
    with pytest.raises(SystemExit, match="records indexed in between"):
        plan_index_window(space, ledger, date(2026, 9, 10), "2026-09-05")
    assert plan_index_window(space, ledger, date(2026, 9, 10), "2026-08-27")[0] == "2026-08-27"


def test_plan_refuses_before_any_fetch_or_with_a_never_fetched_year(tmp_path):
    space = _space(tmp_path, coverage_start="2024-01-01")
    ledger = CoverageLedger(tmp_path / "ledger.json")
    with pytest.raises(SystemExit, match="nothing has been fetched"):
        plan_index_window(space, ledger, date(2026, 9, 10), "last")
    ledger.record_window(space, "2026-01-01", "2026-09-03", fetched=1, path="x")
    with pytest.raises(SystemExit, match="years never fetched"):
        plan_index_window(space, ledger, date(2026, 9, 10), "last")


def test_a_window_short_of_europe_pmcs_count_is_refused():
    from fetch_search_space import check_complete
    check_complete(191_536, 191_536, "2026")
    check_complete(191_500, 191_536, "2026")          # within tolerance
    check_complete(192_000, 191_536, "2026")          # indexed while fetching: more is fine
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        check_complete(148_815, 191_536, "2026")
