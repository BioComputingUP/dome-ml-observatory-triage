import pandas as pd

from dome_triage.pipeline.steps import dedupe_bulk_match_batch


def _row(pmcid=None, pmid=None, doi=None, title="T"):
    return {"pmcid": pmcid, "pmid": pmid, "doi": doi, "title": title}


def test_dedupes_real_sharing_pmid():
    df = pd.DataFrame([_row(pmid="111", title="A"), _row(pmid="111", title="A duplicate")])
    result = dedupe_bulk_match_batch(df)
    assert len(result) == 1


def test_does_not_collapse_distinct_records_missing_pmid():
    # Real, confirmed bug this guards against: pandas drop_duplicates() treats NaN as equal to
    # NaN, so a naive `subset=["pmid"]` dedupe used to collapse every pmid-less record into one
    # survivor, even though these are genuinely different papers.
    df = pd.DataFrame(
        [
            _row(pmid=None, doi="10.1/a", title="Paper A"),
            _row(pmid=None, doi="10.1/b", title="Paper B"),
            _row(pmid=None, doi=None, title="Paper C, no id at all"),
        ]
    )
    result = dedupe_bulk_match_batch(df)
    assert len(result) == 3


def test_dedupes_by_doi_when_pmid_is_missing():
    df = pd.DataFrame(
        [
            _row(pmid=None, doi="10.1/same", title="First copy"),
            _row(pmid=None, doi="10.1/same", title="Second copy, same DOI"),
        ]
    )
    result = dedupe_bulk_match_batch(df)
    assert len(result) == 1


def test_dedupes_by_pmcid_first_when_pmcid_and_doi_disagree():
    # pmcid outranks doi in the established priority (pmcid -> doi -> pmid) -- same real record
    # under two different DOI strings (e.g. a preprint DOI vs. the published version's DOI) but
    # the same PMCID should still collapse to one.
    df = pd.DataFrame(
        [
            _row(pmcid="PMC1", doi="10.1/preprint", title="Preprint version"),
            _row(pmcid="PMC1", doi="10.1/published", title="Published version"),
        ]
    )
    result = dedupe_bulk_match_batch(df)
    assert len(result) == 1


def test_records_with_no_id_at_all_are_never_merged_with_each_other():
    df = pd.DataFrame([_row(title="Unidentified A"), _row(title="Unidentified B")])
    result = dedupe_bulk_match_batch(df)
    assert len(result) == 2


def test_empty_dataframe_returns_empty_without_error():
    df = pd.DataFrame(columns=["pmcid", "pmid", "doi", "title"])
    result = dedupe_bulk_match_batch(df)
    assert result.empty


def test_zero_column_empty_dataframe_does_not_crash():
    # Real, confirmed bug this guards against: raw_records_to_dataframe([]) on a genuinely
    # zero-hit year returns a DataFrame with ZERO COLUMNS (not just zero rows) -- must not KeyError
    # trying to read pmcid/pmid/doi off it.
    df = pd.DataFrame([])
    result = dedupe_bulk_match_batch(df)
    assert result.empty


def test_no_dedup_id_column_leaks_into_the_result():
    df = pd.DataFrame([_row(pmid="111"), _row(pmid="222")])
    result = dedupe_bulk_match_batch(df)
    assert "_dedup_id" not in result.columns
