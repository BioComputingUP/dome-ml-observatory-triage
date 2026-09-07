"""Step 23a(b): consolidates `landscape_classification_events.csv` (one row per DeepSeek call,
including retries) into one row per unique paper, then merges in the real metadata (title,
abstract, pmid/pmcid/doi, year, authors, ...) that already exists in `bulk_candidates.csv` -- no
new fetches, nothing pulled that isn't already on disk.

Named deliberately NOT "canonical" -- that word already means something specific and different in
this project (`canonical_dataset.csv`, the human-curated + second-curator-consensus trusted set).
This is `ai_ml_landscape_classified.csv`: DeepSeek-only, single-pass, one confidence tier.

Memory-safe by construction, reusing `ingest/metadata_repair.py`'s proven pattern: the pool is
streamed in chunks (never loaded whole), and the final file is written to a sibling `.tmp.csv` then
atomically `os.replace`d -- this project has hit real OOM incidents this exact phase from loading
the full 1.8GB pool plus a full in-memory copy at once (`--limit 1000000`, 2026-08-27).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dome_triage.dedupe.keys import record_id_from_ids
from dome_triage.reporting.landscape_profile import resolve_latest_per_record

_READ_CHUNK_ROWS = 100_000

# Explicitly excludes the pool's own placeholder label/label_confidence/source_name/source_file --
# those are "unlabeled"/"unscored" for every row (pre-dating this classification entirely) and
# would sit confusingly next to the real `classification` column from the event log.
_METADATA_COLUMNS_IN_ORDER = [
    "pmid", "pmcid", "doi", "title", "abstract", "year", "authors",
    "journal", "citation_count", "match_metadata", "mesh_headings", "pub_types",
    "is_open_access", "keywords_author", "fulltext_available", "fulltext_source_root",
    "abstract_source", "metadata_repair_sources",
]
_CLASSIFICATION_COLUMNS_IN_ORDER = [
    "classification", "rationale", "model_tier", "mode", "prompt_version",
    "criteria_sha256", "batch_id", "timestamp",
]


def resolve_landscape_classifications(events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last-event-wins per record_id, split into (resolved, unresolved_parse_error). `unresolved`
    holds any record_id whose ONLY-EVER event across the whole file is `parse_error` -- these have
    no real classification to consolidate and must not be silently dropped or silently kept as a
    fake answer."""
    events = pd.read_csv(events_path, dtype=str)
    resolved = resolve_latest_per_record(events)
    unresolved = resolved[resolved["classification"] == "parse_error"].copy()
    resolved_ok = resolved[resolved["classification"] != "parse_error"].copy()
    assert resolved_ok["record_id"].is_unique, "resolve_landscape_classifications: duplicate record_id after resolution"
    return resolved_ok, unresolved


def merge_landscape_metadata(resolved: pd.DataFrame, pool_path: Path, output_path: Path) -> dict:
    """Streams `bulk_candidates.csv` in chunks, keeping only rows whose computed record_id is in
    `resolved`, and writes the merged result (metadata columns first, then the real classification
    columns) to `output_path` via a temp file + atomic replace. Never holds the full merged frame
    in memory -- accumulates matched pool rows (title/abstract text only, no duplication of the
    825k-row classification side) and appends classification-side data per chunk at write time.

    **Real correctness requirement, not just a nice-to-have**: `bulk_candidates.csv` itself still
    holds ~9,050 rows sharing an identity with another row (833,281 unique ids from 842,331 rows,
    confirmed live) -- these are genuine duplicate paper entries (e.g. a preprint and its published
    version collapsing to the same id under the pmcid->doi->pmid priority), never fully collapsed
    at the pool-build stage. Without a guard, matching against `wanted` naively could write MORE
    THAN ONE row for the same record_id, breaking the "one row per unique paper" requirement this
    file exists to satisfy. A `seen` set closes this across chunk boundaries too, not just within
    one chunk -- first occurrence in file order wins, matching this project's established
    dedupe convention (`dedupe_bulk_match_batch`'s `keep="first"`)."""
    wanted = set(resolved["record_id"])
    seen: set[str] = set()
    classification_by_id = resolved.set_index("record_id")[_CLASSIFICATION_COLUMNS_IN_ORDER]

    tmp_path = output_path.with_suffix(".tmp.csv")
    header = True
    n_matched = 0
    n_pool_rows = 0
    n_duplicate_pool_rows_skipped = 0

    read_cols = [
        "pmcid", "pmid", "doi", "title", "abstract", "journal", "authors", "year",
        "citation_count", "match_metadata", "mesh_headings", "pub_types", "is_open_access",
        "keywords_author", "fulltext_available", "fulltext_source_root",
        "abstract_source", "metadata_repair_sources",
    ]
    with tqdm(desc="merging landscape metadata", unit="row") as bar:
        for chunk in pd.read_csv(pool_path, dtype=str, usecols=read_cols, chunksize=_READ_CHUNK_ROWS):
            n_pool_rows += len(chunk)
            id_cols = chunk[["pmcid", "pmid", "doi"]].where(pd.notna(chunk[["pmcid", "pmid", "doi"]]), None)
            chunk = chunk.assign(
                record_id=id_cols.apply(
                    lambda r: record_id_from_ids(r["pmcid"], r["pmid"], r["doi"]), axis=1
                )
            )
            hit = chunk[chunk["record_id"].isin(wanted) & ~chunk["record_id"].isin(seen)]
            if len(hit):
                dupes_within_chunk = hit["record_id"].duplicated(keep="first")
                n_duplicate_pool_rows_skipped += int(dupes_within_chunk.sum())
                hit = hit[~dupes_within_chunk]

                joined = hit.merge(
                    classification_by_id, left_on="record_id", right_index=True, how="inner"
                )
                seen.update(joined["record_id"])  # before the column slice drops it
                joined = joined[_METADATA_COLUMNS_IN_ORDER + _CLASSIFICATION_COLUMNS_IN_ORDER]
                joined.to_csv(tmp_path, mode="w" if header else "a", header=header, index=False)
                header = False
                n_matched += len(joined)
            bar.update(len(chunk))

    if header:
        # Nothing ever matched -- write an empty file with the right header rather than leaving no
        # output at all, so a downstream read never mistakes "missing file" for "zero results".
        pd.DataFrame(columns=_METADATA_COLUMNS_IN_ORDER + _CLASSIFICATION_COLUMNS_IN_ORDER).to_csv(
            tmp_path, index=False
        )

    os.replace(tmp_path, output_path)
    return {
        "n_pool_rows_scanned": n_pool_rows,
        "n_wanted": len(wanted),
        "n_duplicate_pool_rows_skipped": n_duplicate_pool_rows_skipped,
        "n_matched": n_matched,
        "n_unmatched": len(wanted) - n_matched,
    }
