"""Stratified sampling over the scored bulk candidate pool: match-score band x journal bucket x
year bucket, capped per stratum (configs/sampling.yaml) -- builds a diverse, sized-to-fit
candidate queue for the curation app rather than a raw dump of everything above a score cutoff.
`cap_per_stratum` is the one knob that controls total curation workload; see ROADMAP.md's
"Curation workload estimate" section for worked numbers.
"""

from __future__ import annotations

import pandas as pd


def build_strata(
    df: pd.DataFrame,
    score_col: str | None,
    n_score_bands: int = 4,
    top_n_journals: int = 15,
    year_bucket_width: int = 5,
) -> pd.DataFrame:
    """`score_col=None` skips the score-band column entirely -- for pools with no meaningful
    score to band by (e.g. clear-negative candidates, which are explicitly selected for *not*
    matching the lexicon at all, so a lexicon-derived band would be meaningless). Journal/year
    bucketing applies either way."""
    df = df.copy()
    if score_col is not None:
        band_col = f"match_score_band__{score_col}"
        df[band_col] = pd.qcut(df[score_col], q=n_score_bands, labels=False, duplicates="drop")

    top_journals = df["journal"].value_counts().head(top_n_journals).index
    df["journal_bucket"] = df["journal"].where(df["journal"].isin(top_journals), "other")

    # pd.to_numeric first, not a direct .astype("Int64") -- real data has string-formatted-float
    # years ("2012.0", a pre-existing artifact from the dedupe/consolidate pipeline's own CSV
    # round-tripping, confirmed present in canonical_dataset.csv), which .astype("Int64") rejects
    # outright (ArrowInvalid: "Failed to parse string: '2012.0' as a scalar of type int64") --
    # a real crash hit live against real data, not a hypothetical. to_numeric parses both "2012"
    # and "2012.0" fine; the subsequent .astype("Int64") only ever sees clean float/NaN input.
    df["year_bucket"] = (
        pd.to_numeric(df["year"], errors="coerce").astype("Int64") // year_bucket_width * year_bucket_width
    )

    return df


def stratified_sample(
    df: pd.DataFrame,
    strata_cols: list[str],
    cap_per_stratum: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sampled_df, stratum_report_df). The report has one row per stratum combination
    with `available` vs `sampled` counts, so the diversity actually achieved is visible."""
    groups = df.groupby(strata_cols, dropna=False)

    sampled_parts = [
        group.sample(n=min(len(group), cap_per_stratum), random_state=random_state)
        for _, group in groups
    ]
    sampled = pd.concat(sampled_parts).reset_index(drop=True) if sampled_parts else df.iloc[0:0]

    report = groups.size().reset_index(name="available")
    report["sampled"] = report["available"].clip(upper=cap_per_stratum)

    return sampled, report
