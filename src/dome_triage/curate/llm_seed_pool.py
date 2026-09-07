"""Loads the Step 19c LLM/BERT/agentic/bio-language-model seed candidate pool into a DataFrame
shaped for `CurationSession` -- the exact same `record_id`-computation pattern `bulk_pool.py::
load_bulk_pool` uses for the much larger AI/ML bulk pool (see that module's docstring for the full
reasoning), just without its performance-driven column pruning: this pool is ~192 rows, not
~745k, so reading every column costs nothing worth optimizing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dome_triage.dedupe.keys import record_id_from_ids


def load_llm_seed_pool(pool_path: Path) -> pd.DataFrame:
    """Reads `ingest fetch-llm-seed-pool`'s output (full EPMC metadata for every PMID in the Step
    19c seed file, `RawRecord`-shaped via `raw_records_to_dataframe`) and adds `record_id`
    (`dedupe.keys.record_id_from_ids` -- the exact hash a real `dedupe consolidate` run would
    produce for the same pmcid/pmid/doi) so each row is directly usable as a
    `CurationSession.dataset` row with no join back to `canonical_dataset.csv` needed. A record
    that's later merged into `canonical_dataset.csv` gets the identical `record_id` either way, so
    a decision made here lands on the same row once `curate materialize-llm-seed-review` runs --
    same mechanism `state.py::materialize_events`'s `bulk_pool_path` fallback already relies on for
    the main bulk pool."""
    df = pd.read_csv(pool_path, dtype=str)
    df[["pmcid", "pmid", "doi"]] = df[["pmcid", "pmid", "doi"]].fillna("")
    df["record_id"] = [
        record_id_from_ids(pmcid, pmid, doi)
        for pmcid, pmid, doi in zip(df["pmcid"], df["pmid"], df["doi"])
    ]
    return df
