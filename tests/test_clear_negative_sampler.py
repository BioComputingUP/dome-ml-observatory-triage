import csv

import pandas as pd

from dome_triage.ingest.clear_negative_sampler import (
    COMMON_ML_METHOD_TERMS,
    CORE_AI_ML_TERMS,
    fetch_clear_negatives,
    fetch_filtered_clear_negatives,
    select_diversified_pool,
    select_strong_negatives,
)


def _fake_result(idx: int, journal: str, year: int) -> dict:
    return {
        "pmid": str(1000 + idx),
        "pmcid": None,
        "doi": None,
        "title": f"Paper {idx}",
        "abstractText": "An abstract with no AI/ML mention.",
        "authorString": "Someone A.",
        "journalInfo": {"journal": {"title": journal}},
        "pubYear": str(year),
        "isOpenAccess": "N",
        "pubTypeList": {"pubType": ["research-article"]},
        "keywordList": {"keyword": []},
        "inEPMC": "N",
        "meshHeadingList": {"meshHeading": []},
    }


class _FakeSearchClient:
    """Mirrors EpmcClient.search()/.get_by_ids()'s signatures -- returns the same fixed batch for
    every .search() call and looks .get_by_ids() up against that same batch, matching this
    project's existing test pattern (_FakeCountClient in test_bulk_match.py). `fetch_clear_
    negatives` is two-phase now (Phase 1: .search(resultType=lite), Phase 2: .get_by_ids() for the
    winners) -- this fake supports both against one shared fixture list, since real EPMC's lite and
    core results describe the same underlying record."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.queries_seen: list[str] = []
        self.result_types_seen: list[str] = []
        self.get_by_ids_calls: list[tuple[str, list[str]]] = []

    def search(self, query, result_type="core", page_size=None, show_progress=False):
        self.queries_seen.append(query)
        self.result_types_seen.append(result_type)
        return list(self.results)

    def get_by_ids(self, ids: list[str], id_type: str) -> dict[str, dict]:
        self.get_by_ids_calls.append((id_type, list(ids)))
        ids_set = set(ids)
        return {r[id_type]: r for r in self.results if r.get(id_type) in ids_set}


def _diverse_results(n_per_journal: int = 5) -> list[dict]:
    journals_years = [
        ("Journal A", 2010),
        ("Journal B", 2015),
        ("Journal C", 2020),
        ("Journal D", 2005),
    ]
    results = []
    idx = 0
    for journal, year in journals_years:
        for _ in range(n_per_journal):
            results.append(_fake_result(idx, journal, year))
            idx += 1
    return results


def test_fetch_clear_negatives_returns_raw_pool_when_below_sample_size():
    client = _FakeSearchClient(_diverse_results(n_per_journal=2))  # 8 raw records
    df = fetch_clear_negatives(client, 2000, 2020, sample_size=100, n_windows=1)

    assert len(df) == 8
    assert "journal_bucket" not in df.columns
    assert "year_bucket" not in df.columns
    assert set(df["label"]) == {"negative"}
    assert set(df["label_confidence"]) == {"heuristic_candidate"}
    assert set(df["source_name"]) == {"clear_negative_sampler"}


def test_fetch_clear_negatives_stratifies_and_caps_when_above_sample_size():
    client = _FakeSearchClient(_diverse_results(n_per_journal=10))  # 40 raw records, 4 journals

    df = fetch_clear_negatives(client, 2000, 2020, sample_size=12, n_windows=1, top_n_journals=4)

    assert len(df) <= 12
    assert "journal_bucket" not in df.columns  # strata columns never leak into the return value
    # stratification must actually spread across journals, not just take the first 12 raw rows
    # (which would all be "Journal A" given _diverse_results' construction order).
    assert df["journal"].nunique() > 1


def test_fetch_clear_negatives_calls_search_once_per_window():
    client = _FakeSearchClient(_diverse_results(n_per_journal=1))
    fetch_clear_negatives(client, 2000, 2001, sample_size=100, n_windows=3)

    assert len(client.queries_seen) == 3
    for query in client.queries_seen:
        assert "NOT" in query
        assert "artificial intelligence" in query


def test_exclude_query_covers_core_ai_ml_terms_and_common_ml_methods():
    client = _FakeSearchClient(_diverse_results(n_per_journal=1))
    fetch_clear_negatives(client, 2000, 2001, sample_size=100, n_windows=1)

    query = client.queries_seen[0]
    for term in CORE_AI_ML_TERMS + COMMON_ML_METHOD_TERMS:
        assert f'"{term}"' in query, f"expected {term!r} in the exclude query"


def test_fetch_clear_negatives_uses_lite_for_phase_1_and_get_by_ids_for_phase_2():
    # Phase 1 must request the small resultType=lite payload, not full core records, for every
    # window -- this is the whole point of the two-phase rewrite (see the module's performance
    # reasoning). Phase 2 fetches full records only for the diversified winners via get_by_ids.
    client = _FakeSearchClient(_diverse_results(n_per_journal=1))
    fetch_clear_negatives(client, 2000, 2001, sample_size=100, n_windows=1)

    assert client.result_types_seen == ["lite"]
    assert len(client.get_by_ids_calls) >= 1
    id_type, ids = client.get_by_ids_calls[0]
    assert id_type == "pmid"
    assert set(ids) == {str(1000 + i) for i in range(4)}  # 4 journals, n_per_journal=1


def test_fetch_clear_negatives_caps_per_window_via_early_break():
    # 100 fixture results, all from one window -- max_per_window must stop Phase 1 from collecting
    # more than the cap, even though the fake .search() hands back the whole list in one call (the
    # real EpmcClient.search() would instead stop issuing further paginated HTTP requests once the
    # caller stops iterating -- this is what actually avoids trawling through every match).
    results = [_fake_result(i, "Journal A", 2015) for i in range(100)]
    client = _FakeSearchClient(results)

    df = fetch_clear_negatives(client, 2015, 2015, sample_size=1000, n_windows=1, max_per_window=10)

    assert len(df) == 10


def _screened_row(idx: int, journal: str, year: int, needs_screening: bool) -> dict:
    return {
        "record_id": f"r{idx}",
        "pmid": str(1000 + idx),
        "title": f"Paper {idx}",
        "abstract": "An abstract with no AI/ML mention.",
        "journal": journal,
        "year": year,
        "lexicon_score__bm25": 5.0 if not needs_screening else 500.0,
        "needs_screening": needs_screening,
    }


def test_select_strong_negatives_does_not_exclude_flagged_rows():
    df = pd.DataFrame(
        [
            _screened_row(0, "Journal A", 2010, needs_screening=False),
            _screened_row(1, "Journal A", 2010, needs_screening=True),
            _screened_row(2, "Journal B", 2015, needs_screening=False),
        ]
    )

    selected, n_flagged = select_strong_negatives(df, limit=10)

    # under the limit -> the full pool is kept, flagged rows included, not dropped
    assert set(selected["record_id"]) == {"r0", "r1", "r2"}
    assert n_flagged == 1  # informational count within the selected batch, not an exclusion
    assert "journal_bucket" not in selected.columns
    assert "year_bucket" not in selected.columns


def test_select_strong_negatives_redivsersifies_and_caps_when_over_limit():
    journals_years = [("Journal A", 2010), ("Journal B", 2015), ("Journal C", 2020)]
    rows = []
    idx = 0
    for journal, year in journals_years:
        for _ in range(10):
            rows.append(_screened_row(idx, journal, year, needs_screening=False))
            idx += 1
    df = pd.DataFrame(rows)

    selected, n_flagged = select_strong_negatives(df, limit=9, top_n_journals=3)

    assert n_flagged == 0
    assert len(selected) <= 9
    assert selected["journal"].nunique() > 1


def test_select_strong_negatives_can_include_flagged_rows_even_when_over_limit():
    # A pool entirely flagged needs_screening=True must still be selectable -- the flag is
    # diagnostic-only and must never act as a hidden filter, even under the re-diversify/cap path.
    journals_years = [("Journal A", 2010), ("Journal B", 2015), ("Journal C", 2020)]
    rows = []
    idx = 0
    for journal, year in journals_years:
        for _ in range(10):
            rows.append(_screened_row(idx, journal, year, needs_screening=True))
            idx += 1
    df = pd.DataFrame(rows)

    selected, n_flagged = select_strong_negatives(df, limit=9, top_n_journals=3)

    assert len(selected) == 9
    assert n_flagged == 9  # all of the (small) selected batch happen to be flagged -- fine


# ---------------------------------------------------------------------------
# select_diversified_pool -- the extracted re-diversify/cap logic (Step 19d)
# ---------------------------------------------------------------------------


def _plain_row(idx: int, journal: str, year: int) -> dict:
    return {"record_id": f"r{idx}", "journal": journal, "year": year}


def test_select_diversified_pool_returns_unchanged_when_under_limit():
    df = pd.DataFrame([_plain_row(0, "Journal A", 2010), _plain_row(1, "Journal B", 2015)])

    result = select_diversified_pool(df, limit=10)

    assert set(result["record_id"]) == {"r0", "r1"}
    assert "journal_bucket" not in result.columns


def test_select_diversified_pool_redivsersifies_and_caps_when_over_limit():
    journals_years = [("Journal A", 2010), ("Journal B", 2015), ("Journal C", 2020)]
    rows = [_plain_row(idx, j, y) for idx, (j, y) in enumerate(jy for jy in journals_years for _ in range(10))]
    df = pd.DataFrame(rows)

    result = select_diversified_pool(df, limit=9, top_n_journals=3)

    assert len(result) <= 9
    assert result["journal"].nunique() > 1


def test_select_diversified_pool_deterministic_for_fixed_seed():
    journals_years = [("Journal A", 2010), ("Journal B", 2015), ("Journal C", 2020)]
    rows = [_plain_row(idx, j, y) for idx, (j, y) in enumerate(jy for jy in journals_years for _ in range(10))]
    df = pd.DataFrame(rows)

    first = select_diversified_pool(df, limit=9, top_n_journals=3)
    second = select_diversified_pool(df, limit=9, top_n_journals=3)

    assert sorted(first["record_id"]) == sorted(second["record_id"])


# ---------------------------------------------------------------------------
# fetch_filtered_clear_negatives -- the hard review/non-methods exclusion gate (Step 19d)
# ---------------------------------------------------------------------------


def _write_lexicon(tmp_path) -> str:
    path = tmp_path / "exclusionary.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "discriminative_score", "document_frequency", "source", "notes"])
        writer.writerow(["review", "", "", "test", "non_methods_pubtype"])
        writer.writerow(["systematic review", "", "", "test", "non_methods_pubtype"])
    return str(path)


def _fake_result_with_pubtype(idx: int, journal: str, year: int, pub_types: list[str], abstract: str) -> dict:
    return {
        "pmid": str(2000 + idx),
        "pmcid": None,
        "doi": None,
        "title": f"Paper {idx}",
        "abstractText": abstract,
        "authorString": "Someone A.",
        "journalInfo": {"journal": {"title": journal}},
        "pubYear": str(year),
        "isOpenAccess": "N",
        "pubTypeList": {"pubType": pub_types},
        "keywordList": {"keyword": []},
        "inEPMC": "N",
        "meshHeadingList": {"meshHeading": []},
    }


def test_fetch_filtered_clear_negatives_drops_flagged_rows(tmp_path):
    results = [
        _fake_result_with_pubtype(0, "Journal A", 2010, ["research-article"], "A primary study of gene expression."),
        _fake_result_with_pubtype(1, "Journal B", 2011, ["Review"], "A broad review of the field."),
        _fake_result_with_pubtype(
            2, "Journal C", 2012, ["research-article"], "We performed a systematic review of trials."
        ),
        _fake_result_with_pubtype(3, "Journal D", 2013, ["research-article"], "A second primary study."),
    ]
    client = _FakeSearchClient(results)
    lexicon_path = _write_lexicon(tmp_path)

    selected, stats = fetch_filtered_clear_negatives(
        client, 2000, 2020, raw_pool_size=100, target_size=10, exclusionary_lexicon_path=lexicon_path, n_windows=1
    )

    assert stats["fetched"] == 4
    assert stats["dropped_by_gate"] == 2  # pmid 2001 (pub_type Review) + pmid 2002 (text "systematic review")
    assert stats["survivors"] == 2
    assert stats["selected"] == 2
    assert stats["shortfall"] == 8
    assert set(selected["pmid"]) == {"2000", "2003"}


def test_fetch_filtered_clear_negatives_reports_shortfall_without_raising(tmp_path):
    # Every candidate is flagged -- selected must come back empty, not raise, with shortfall
    # equal to the full target_size.
    results = [_fake_result_with_pubtype(0, "Journal A", 2010, ["Review"], "A broad review of the field.")]
    client = _FakeSearchClient(results)
    lexicon_path = _write_lexicon(tmp_path)

    selected, stats = fetch_filtered_clear_negatives(
        client, 2000, 2020, raw_pool_size=100, target_size=5, exclusionary_lexicon_path=lexicon_path, n_windows=1
    )

    assert len(selected) == 0
    assert stats["shortfall"] == 5
