"""Loads `bulk_candidates_scored.csv` (the full ~745k-record Europe PMC AI/ML pool, Step 12) into
a DataFrame shaped exactly like `CurationSession` expects `dataset` to be -- so a curator can
browse and decide records directly from the full bulk pool, ranked by BM25 score, instead of only
the pre-stratified queue Step 13 sampled into `canonical_dataset.csv`. See `CurationSession`'s
`dataset_df` parameter (state.py) for the injection point this feeds.

Deliberately does NOT import from `pipeline/steps.py` (pulls in `torch`), matching every other
module in this package -- see `bulk_scores.py`'s module docstring for the same reasoning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dome_triage.dedupe.keys import record_id_from_ids

_USECOLS = [
    "source_name",
    "label",
    "label_confidence",
    "pmcid",
    "pmid",
    "doi",
    "title",
    "abstract",
    "journal",
    "year",
    "mesh_headings",
    "already_curated",
    "existing_label",
    "existing_label_confidence",
    "match_score__bm25",
    "match_classification__bm25",
]
# `authors`/`pub_types` were dropped from this list after a real, measured incident: with them
# included, `pd.read_csv`'s C engine peaked at ~5.6GB RSS parsing this ~745k-row file even though
# the resulting DataFrame's own "deep" memory was only ~1.5GB -- a live host OOM-kill (confirmed
# via `journalctl -k`, `oom-kill: ... task=python ... anon-rss:8681148kB`, the combined RSS of
# this frame plus the separately-cached bulk_scores.py score lookup, both held by the same
# Streamlit process after switching between the Curate page's two queue sources). Dropping just
# these two unused-for-display columns cut peak RSS to ~1.9GB -- disproportionate to their own
# ~90MB of final column data, which points at the C engine's per-row tokenization/buffering
# overhead scaling non-linearly with column count on a file this wide, not at the final data size.
# Neither column is read by the Curate page (title/abstract/journal/year/mesh_headings/pmcid/pmid/
# doi/score/classification are); if a future caller needs them for display, re-measure RSS before
# adding them back rather than assuming a couple of extra columns are free.


def load_bulk_pool(scored_path: Path) -> pd.DataFrame:
    """Reads only the columns `CurationSession`/the Curate page actually touch (title, abstract,
    journal, year, mesh_headings, the three id fields, label/confidence, BM25 score) -- not the
    file's other ~9 columns (source_file, match_metadata, fulltext_source_root, has_pmcid,
    matched_terms__bm25) that this browsing mode has no use for. Still a real, multi-hundred-MB
    one-time read at 744,647 rows; cache it (see streamlit_helpers.py's `_cached_bulk_pool`,
    `st.cache_resource`, mtime-keyed) exactly like `bulk_scores.py`'s score-only lookup -- never
    re-read per Streamlit rerun.

    Two things make each row usable as a `CurationSession.dataset` row on its own, with no join
    back to `canonical_dataset.csv` needed:

    1. **`record_id`** is computed here via `dedupe.keys.record_id_from_ids` -- the *exact* same
       `sha1(canonical_key)` a real `dedupe consolidate` run would produce for this record's
       pmcid/pmid/doi. A record already merged into `canonical_dataset.csv` therefore gets the
       identical record_id whether reached from there or from here, so an event logged against it
       from either source lands on the same row once `curate materialize` runs (see
       `state.py::materialize_events`'s `bulk_pool_path` fallback for the case where it hasn't
       been merged in yet at all).
    2. **`label`/`label_confidence`** are taken from `existing_label`/`existing_label_confidence`
       when `already_curated` is True (Step 12's `_annotate_already_curated` join -- a prior
       curation round already decided this record) and from the raw `label`/`label_confidence`
       columns otherwise (always `"unlabeled"`/`"unscored"` for a fresh bulk-match candidate, per
       `bulk_match.py::core_result_to_raw_record`'s defaults). This is what makes
       `CurationSession`'s existing trusted-label exclusion logic
       (`_TRUSTED_LABEL_CONFIDENCE`/`include_already_labeled`) work correctly here without any
       bulk-pool-specific branching in `state.py` -- an already-curated bulk-pool row looks, to
       that logic, exactly like the equivalent already-curated canonical_dataset row.

    `match_score__bm25`/`match_classification__bm25` are renamed to `bulk_match_score`/
    `bulk_match_classification` -- the column names `CurationSession`/the page already read,
    matching what `bulk_scores.py::annotate_bulk_scores` would have produced via a join. Renaming
    once here means a bulk-pool session passes `bulk_score_lookup=None` and skips that join
    entirely -- the score is already native to this file, a second lookup-join would just
    recompute what's already sitting in the column.
    """
    df = pd.read_csv(scored_path, usecols=_USECOLS, dtype=str)
    # `pd.read_csv(..., dtype=str)` still represents a genuinely empty id cell as float `nan`, not
    # `""` -- `record_id_from_ids`/`canonical_key_from_ids` check id presence via a plain `if
    # value:`, and `nan` is truthy in Python, so an unfilled empty pmcid could be treated as a
    # *present* id (producing a wrong "PMCID:nan" key) instead of correctly falling through to
    # doi/pmid. fillna here, before the id columns are used for anything, is what makes that
    # priority fallback behave correctly.
    df[["pmcid", "pmid", "doi"]] = df[["pmcid", "pmid", "doi"]].fillna("")

    df["record_id"] = [
        record_id_from_ids(pmcid, pmid, doi)
        for pmcid, pmid, doi in zip(df["pmcid"], df["pmid"], df["doi"])
    ]
    # A record with no pmcid/pmid/doi at all (real, rare -- exactly 1 of 744,647 in the live pool)
    # has no computable record_id. The natural fix, `df[df["record_id"].notna()].reset_index
    # (drop=True)`, is a real, measured incident: boolean-mask row selection on this wide,
    # object-dtype-heavy frame forces pandas to copy essentially the entire ~1.9GB DataFrame to
    # drop a single row -- confirmed via `resource.getrusage().ru_maxrss` climbing to ~5.5GB for
    # that one line alone, and `del`+`gc.collect()` afterward does NOT reclaim it (glibc doesn't
    # return freed heap arenas to the OS). A synthetic, uniquely-numbered placeholder id costs one
    # cheap Series-level fillna instead of a whole-frame copy -- this row still can't be reconciled
    # with a real `dedupe consolidate` output (there's no id to reconcile from), but it stays
    # browsable/decidable rather than needing to vanish, and `materialize_events`'s bulk-pool
    # insert path already warns rather than silently drops if this placeholder is ever decided and
    # can't be found again by its real (nonexistent) ids.
    missing = df["record_id"].isna()
    if missing.any():
        df.loc[missing, "record_id"] = [f"bulkpool-no-id-row-{i}" for i in df.index[missing]]

    already_curated = df["already_curated"].astype(str).str.lower() == "true"
    df["label"] = df["existing_label"].where(already_curated, df["label"])
    df["label_confidence"] = df["existing_label_confidence"].where(already_curated, df["label_confidence"])

    df = df.rename(
        columns={
            "match_score__bm25": "bulk_match_score",
            "match_classification__bm25": "bulk_match_classification",
        }
    )
    df["bulk_match_score"] = pd.to_numeric(df["bulk_match_score"], errors="coerce")
    return df
