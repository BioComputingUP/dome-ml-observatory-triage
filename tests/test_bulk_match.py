import pytest

from dome_triage.ingest import bulk_match
from dome_triage.ingest.bulk_match import (
    _range_query,
    core_result_to_raw_record,
    count_ai_ml_breakdown,
    fetch_ai_ml_range,
)

# Field structure confirmed via a live Europe PMC resultType=core fetch.
SAMPLE_RESULT = {
    "pmid": "35582656",
    "pmcid": "PMC8962827",
    "doi": "10.1109/tim.2021.3130675",
    "title": "Deep Generative Learning-Based 1-SVM Detectors.",
    "abstractText": "A sample blood test has recently become an important tool.",
    "authorString": "Dairi A, Harrou F, Sun Y.",
    "journalInfo": {"journal": {"title": "IEEE transactions on instrumentation and measurement"}},
    "pubYear": "2022",
    "isOpenAccess": "Y",
    "pubTypeList": {"pubType": ["research-article", "Journal Article"]},
    "keywordList": {"keyword": ["Deep Learning", "Covid-19"]},
    "inEPMC": "Y",
    "meshHeadingList": {
        "meshHeading": [{"majorTopic_YN": "Y", "descriptorName": "Machine Learning"}]
    },
}


def test_core_result_to_raw_record_maps_all_fields():
    record = core_result_to_raw_record(SAMPLE_RESULT, source_name="bulk_match_2022", source_file="test.jsonl")

    assert record.pmcid == "PMC8962827"
    assert record.pmid == "35582656"
    assert record.doi == "10.1109/tim.2021.3130675"
    assert record.title == "Deep Generative Learning-Based 1-SVM Detectors."
    assert record.journal == "IEEE transactions on instrumentation and measurement"
    assert record.year == 2022
    assert record.is_open_access is True
    assert record.pub_types == ["research-article", "Journal Article"]
    assert record.keywords_author == ["Deep Learning", "Covid-19"]
    assert record.mesh_headings == ["Machine Learning"]
    assert record.fulltext_available is True
    assert record.label == "unlabeled"
    assert record.label_confidence == "unscored"


def test_core_result_to_raw_record_accepts_label_override():
    record = core_result_to_raw_record(
        SAMPLE_RESULT,
        source_name="clear_negative_sampler",
        source_file="test",
        label="negative",
        label_confidence="heuristic_candidate",
    )
    assert record.label == "negative"
    assert record.label_confidence == "heuristic_candidate"


def test_core_result_to_raw_record_handles_missing_optional_fields():
    minimal_result = {"pmid": "12345678", "title": "A paper"}
    record = core_result_to_raw_record(minimal_result, source_name="test", source_file="test")

    assert record.pmid == "12345678"
    assert record.pmcid is None
    assert record.year is None
    assert record.is_open_access is None
    assert record.mesh_headings == []
    assert record.pub_types == []
    assert record.fulltext_available is False


def test_core_result_to_raw_record_filters_null_entries_in_keyword_and_pubtype_lists():
    # Confirmed live in the real 2000-2026 bulk fetch: some records have a bare `null` inside
    # keywordList/pubTypeList, which used to raise a Pydantic ValidationError and abort the whole
    # load rather than just dropping the one bad list element.
    result = dict(
        SAMPLE_RESULT,
        keywordList={"keyword": [None, "Deep Learning", None]},
        pubTypeList={"pubType": ["Journal Article", None]},
    )
    record = core_result_to_raw_record(result, source_name="test", source_file="test.jsonl")

    assert record.keywords_author == ["Deep Learning"]
    assert record.pub_types == ["Journal Article"]


def test_range_query_builds_expected_date_bounded_query():
    query = _range_query(2000, 2026)
    assert query == '("artificial intelligence" OR "machine learning") AND (FIRST_PDATE:[2000-01-01 TO 2026-12-31])'


def test_range_query_accepts_a_single_term_query():
    query = _range_query(2010, 2010, '"machine learning"')
    assert query == '("machine learning") AND (FIRST_PDATE:[2010-01-01 TO 2010-12-31])'


def test_fetch_ai_ml_range_fetches_each_year_in_order(monkeypatch, tmp_path):
    fetched_years = []

    def fake_fetch(client, year, checkpoint_dir):
        fetched_years.append(year)
        return checkpoint_dir / f"bulk_match_{year}.jsonl"

    monkeypatch.setattr(bulk_match, "fetch_ai_ml_candidates", fake_fetch)

    paths = fetch_ai_ml_range(client=object(), year_from=2020, year_to=2023, checkpoint_dir=tmp_path)

    assert fetched_years == [2020, 2021, 2022, 2023]
    assert paths == [tmp_path / f"bulk_match_{y}.jsonl" for y in fetched_years]


def test_fetch_ai_ml_range_rejects_year_to_before_year_from(tmp_path):
    with pytest.raises(ValueError):
        fetch_ai_ml_range(client=object(), year_from=2023, year_to=2020, checkpoint_dir=tmp_path)


class _FakeCountClient:
    def __init__(self, counts_by_query: dict):
        self.counts_by_query = counts_by_query
        self.queries_seen = []

    def count(self, query: str) -> int:
        self.queries_seen.append(query)
        return self.counts_by_query[query]


def test_count_ai_ml_breakdown_queries_ai_ml_and_combined_separately():
    ai_query = '("artificial intelligence") AND (FIRST_PDATE:[2000-01-01 TO 2001-12-31])'
    ml_query = '("machine learning") AND (FIRST_PDATE:[2000-01-01 TO 2001-12-31])'
    combined_query = (
        '("artificial intelligence" OR "machine learning") AND '
        "(FIRST_PDATE:[2000-01-01 TO 2001-12-31])"
    )
    client = _FakeCountClient({ai_query: 100, ml_query: 200, combined_query: 250})

    breakdown = count_ai_ml_breakdown(client, 2000, 2001)

    assert breakdown == {"ai_count": 100, "ml_count": 200, "combined_count": 250}
    # combined must be <= ai + ml in any real dataset (overlap only shrinks it) -- a sanity
    # invariant on the fake data itself, not the function, but worth asserting explicitly.
    assert breakdown["combined_count"] <= breakdown["ai_count"] + breakdown["ml_count"]
