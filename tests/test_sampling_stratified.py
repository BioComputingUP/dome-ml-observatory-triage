import pandas as pd

from dome_triage.sampling.stratified import build_strata, stratified_sample


def _synthetic_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_score": [(i % 10) / 10 for i in range(n)],
            "journal": [["Nature", "Science", "PLOS ONE", "Bioinformatics"][i % 4] for i in range(n)],
            "year": [2010 + (i % 15) for i in range(n)],
        }
    )


def test_build_strata_adds_expected_columns():
    df = _synthetic_df()
    strata_df = build_strata(df, score_col="match_score", n_score_bands=2, top_n_journals=2, year_bucket_width=5)

    assert "match_score_band__match_score" in strata_df.columns
    assert "journal_bucket" in strata_df.columns
    assert "year_bucket" in strata_df.columns
    # only the top 2 journals keep their name; the rest collapse to "other"
    assert set(strata_df["journal_bucket"].unique()) <= {"Nature", "Science", "other"}


def test_stratified_sample_respects_cap_per_stratum():
    df = _synthetic_df()
    strata_df = build_strata(df, score_col="match_score", n_score_bands=2, top_n_journals=4, year_bucket_width=5)
    strata_cols = ["match_score_band__match_score", "journal_bucket", "year_bucket"]

    sampled, report = stratified_sample(strata_df, strata_cols, cap_per_stratum=2)

    counts = sampled.groupby(strata_cols).size()
    assert (counts <= 2).all()
    assert (report["sampled"] <= report["available"]).all()
    assert (report["sampled"] <= 2).all()


def test_stratified_sample_report_reflects_actual_availability():
    df = _synthetic_df(n=8)  # small enough that some strata have fewer than the cap available
    strata_df = build_strata(df, score_col="match_score", n_score_bands=1, top_n_journals=4, year_bucket_width=20)
    strata_cols = ["match_score_band__match_score", "journal_bucket", "year_bucket"]

    _, report = stratified_sample(strata_df, strata_cols, cap_per_stratum=100)
    # cap far exceeds availability -- sampled should equal available, not the cap
    assert (report["sampled"] == report["available"]).all()


def test_build_strata_with_score_col_none_skips_score_band_column():
    # clear-negative candidates have no meaningful lexicon score to band by -- score_col=None
    # must skip that column entirely rather than erroring, while journal/year bucketing still work.
    df = _synthetic_df()
    strata_df = build_strata(df, score_col=None, top_n_journals=2, year_bucket_width=5)

    assert not any(col.startswith("match_score_band__") for col in strata_df.columns)
    assert "journal_bucket" in strata_df.columns
    assert "year_bucket" in strata_df.columns


def test_stratified_sample_works_with_journal_year_only_strata():
    df = _synthetic_df()
    strata_df = build_strata(df, score_col=None, top_n_journals=4, year_bucket_width=5)
    strata_cols = ["journal_bucket", "year_bucket"]

    sampled, report = stratified_sample(strata_df, strata_cols, cap_per_stratum=2)

    counts = sampled.groupby(strata_cols).size()
    assert (counts <= 2).all()
    assert (report["sampled"] <= report["available"]).all()


def test_build_strata_handles_string_formatted_float_years():
    # Real crash hit live against canonical_dataset.csv: some `year` values are stored as
    # "2012.0" (a pre-existing artifact from the dedupe/consolidate pipeline's CSV round-tripping,
    # not synthetic) -- a direct df["year"].astype("Int64") rejects that string outright
    # (ArrowInvalid). Must parse cleanly via pd.to_numeric first.
    df = pd.DataFrame(
        {
            "match_score": [0.1, 0.2, 0.3],
            "journal": ["Nature", "Science", "Nature"],
            "year": ["2012.0", "2015", None],
        }
    )
    strata_df = build_strata(df, score_col="match_score", n_score_bands=1, top_n_journals=2, year_bucket_width=5)

    assert strata_df["year_bucket"].tolist() == [2010, 2015, pd.NA]
