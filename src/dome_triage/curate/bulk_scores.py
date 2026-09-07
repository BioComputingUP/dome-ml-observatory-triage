"""Joins bulk-pool scoring/screening columns onto canonical/session rows for the Curate app's
filters and diversity dashboard -- the mirror image of pipeline/steps.py's
`_build_existing_label_lookup`/`_annotate_already_curated` (which join canonical *labels* onto the
bulk pool, for score-bulk-match's reporting columns). Here it's the opposite direction: bulk-pool
*scores* joined onto canonical rows, so the Curate app can filter/display them without
`canonical_dataset.csv` itself ever gaining a `match_score__bm25`-style column (Pydantic's
`extra='ignore'` already drops those on merge -- a deliberate choice, not an accident, kept that
way here too).

Deliberately does NOT import from `pipeline/steps.py`, which transitively pulls in `torch` via
`keywords/keybert_extract.py` -- the Streamlit process shouldn't pay that import cost just to
build a lookup dict.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ID_FIELDS = ("pmcid", "pmid", "doi")


def _existing_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in wanted if c in header]


def load_bulk_score_lookup(
    scored_path: Path,
    score_col: str = "match_score__bm25",
    classification_col: str = "match_classification__bm25",
) -> dict[str, tuple[float, str]]:
    """Maps every pmcid/pmid/doi in `bulk_candidates_scored.csv` to (score, classification).
    Reads only the columns needed -- the file is ~1.7GB/27 columns at 744k+ rows; loading
    title/abstract here would be a multi-GB cost just to build a lookup dict. Returns `{}`
    gracefully if `scored_path` doesn't exist (Step 12 hasn't been run yet) or `score_col` is
    missing (e.g. a different scorer was chosen than the caller's default)."""
    scored_path = Path(scored_path)
    if not scored_path.exists():
        return {}

    usecols = _existing_usecols(scored_path, [*_ID_FIELDS, score_col, classification_col])
    if score_col not in usecols:
        return {}
    has_classification = classification_col in usecols

    df = pd.read_csv(scored_path, usecols=usecols, dtype=str)
    scores = pd.to_numeric(df[score_col], errors="coerce")
    classifications = df[classification_col] if has_classification else pd.Series([""] * len(df))

    lookup: dict[str, tuple[float, str]] = {}
    for id_field in _ID_FIELDS:
        if id_field not in df.columns:
            continue
        mask = df[id_field].notna() & (df[id_field] != "")
        for value, score, classification in zip(df.loc[mask, id_field], scores[mask], classifications[mask]):
            lookup.setdefault(value, (score, classification or ""))
    return lookup


def annotate_bulk_scores(
    dataset: pd.DataFrame,
    lookup: dict[str, tuple[float, str]],
    score_out_col: str = "bulk_match_score",
    classification_out_col: str = "bulk_match_classification",
) -> pd.DataFrame:
    """pmcid->pmid->doi fallback join in a single pass over `dataset`'s rows, short-circuiting on
    the first id field that hits. Returns a copy; never writes back to `dataset` or to disk.

    Deliberately does NOT use `Series.map(lookup)`, despite that being the idiomatic-looking
    choice, and this is the second time the same class of bug has been fixed here -- both found by
    live profiling, both worth stating so it doesn't get "cleaned up" back into one:

      1. An earlier version rebuilt `{k: v[0] for k, v in lookup.items()}` (and the same for
         `v[1]`) on *every call*. At `lookup`'s real size -- ~2.07 million entries (3 id fields x
         ~745k bulk-pool rows) -- that unzip alone cost ~9-10s per call.
      2. Replacing that with `Series.map(lookup)` looked vectorized but was still ~2.5s per call,
         because pandas' `map_array` fast-paths a plain dict by doing `Series(mapper)` -- building
         a full 2.07M-element Series *and* Index and hashing every key -- before taking the 2,862
         values actually wanted. Cost scales with the dict, not the Series, which is the opposite
         of what the code reads like. (A dict subclass defining `__missing__` takes a different
         pandas branch, but relying on that is far more obscure than just not calling `.map()`.)

    Iterating `dataset`'s few thousand rows and calling `lookup.get()` is O(rows) with O(1) gets --
    it never touches the other ~2.07M entries at all. This is not the `.iterrows()` pattern this
    codebase avoids at bulk-pool scale: it's a plain zip over 1-3 already-extracted columns of a
    small frame, and it measured ~2.5s -> ~0.01s."""
    dataset = dataset.copy()
    id_cols = [f for f in _ID_FIELDS if f in dataset.columns]
    if not id_cols:
        dataset[score_out_col] = None
        dataset[classification_out_col] = ""
        return dataset

    scores: list = []
    classifications: list = []
    for row_ids in zip(*(dataset[c] for c in id_cols)):
        hit = None
        for value in row_ids:
            hit = lookup.get(value)
            if hit is not None:
                break
        scores.append(hit[0] if hit is not None else None)
        classifications.append(hit[1] if hit is not None else "")

    dataset[score_out_col] = scores
    dataset[classification_out_col] = classifications
    return dataset


def load_screening_lookup(screened_path: Path) -> dict[str, bool]:
    """Maps pmcid/pmid/doi -> `needs_screening` from `clear_negative_candidates_screened.csv`
    (Step 14b). Returns `{}` gracefully if `screened_path` doesn't exist (screening hasn't been
    run) or the column is missing."""
    screened_path = Path(screened_path)
    if not screened_path.exists():
        return {}

    usecols = _existing_usecols(screened_path, [*_ID_FIELDS, "needs_screening"])
    if "needs_screening" not in usecols:
        return {}

    df = pd.read_csv(screened_path, usecols=usecols, dtype=str)
    flags = df["needs_screening"].isin(["True", "true", "1"])

    lookup: dict[str, bool] = {}
    for id_field in _ID_FIELDS:
        if id_field not in df.columns:
            continue
        mask = df[id_field].notna() & (df[id_field] != "")
        for value, flag in zip(df.loc[mask, id_field], flags[mask]):
            lookup.setdefault(value, bool(flag))
    return lookup


def annotate_screening(dataset: pd.DataFrame, lookup: dict[str, bool], out_col: str = "needs_screening") -> pd.DataFrame:
    """Same single-pass fallback-join as `annotate_bulk_scores` (see its docstring for why
    `Series.map(lookup)` is deliberately avoided -- the same pandas cost applies here, and this
    lookup reaches comparable size once Step 14b has run over a full clear-negative batch), for the
    boolean screening flag."""
    dataset = dataset.copy()
    id_cols = [f for f in _ID_FIELDS if f in dataset.columns]
    if not id_cols:
        dataset[out_col] = False
        return dataset

    flags: list = []
    for row_ids in zip(*(dataset[c] for c in id_cols)):
        hit = None
        for value in row_ids:
            hit = lookup.get(value)
            if hit is not None:
                break
        flags.append(bool(hit))

    dataset[out_col] = flags
    return dataset


def load_youden_threshold(
    bakeoff_report_path: Path, scorer_name: str = "bm25", condition: str = "positive_plus_exclusionary_lexicon"
) -> float | None:
    """Reads the Youden-optimal threshold Step 11's bake-off validated for `scorer_name`/
    `condition` from `scoring_bakeoff_report.csv` -- for the Curate app's BM25-score tooltip, so
    the "what does this number mean" explanation cites the *real* validated cutoff, not a
    hardcoded guess. Mirrors `pipeline/steps.py::_lookup_youden_threshold` exactly, duplicated
    (not imported) for the same reason as this module's other functions: importing from
    `pipeline.steps` would pull `torch` into the Streamlit process. Returns None gracefully if the
    report doesn't exist yet or has no matching row."""
    bakeoff_report_path = Path(bakeoff_report_path)
    if not bakeoff_report_path.exists():
        return None
    report = pd.read_csv(bakeoff_report_path)
    match = report[(report["scorer"] == scorer_name) & (report["condition"] == condition)]
    if match.empty or pd.isna(match.iloc[0]["threshold_youden"]):
        return None
    return float(match.iloc[0]["threshold_youden"])
