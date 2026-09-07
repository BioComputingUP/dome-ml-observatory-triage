"""Draws the blind 500+500 (or custom) evaluation sample for Step 20, and defines the actual
blinding boundary (`strip_for_api`) between a full canonical_dataset.csv row and what a DeepSeek
prompt is ever allowed to see.
"""

from __future__ import annotations

import pandas as pd

# Same trusted tier `curate/state.py::CurationSession` already treats as settled ground truth
# elsewhere in this codebase -- drawing from anything less (e.g. heuristic_candidate rows, never
# human-reviewed at all) would not be testing DeepSeek against real human judgment.
_TRUSTED_LABEL_CONFIDENCE = ("human_curated", "registry_confirmed")

_API_ALLOWED_FIELDS = ("title", "abstract", "journal", "year")


def draw_blind_sample(
    dataset: pd.DataFrame,
    n_positive: int = 500,
    n_negative: int = 500,
    random_state: int = 42,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """Plain random draw (per Gavin's confirmed choice -- not oversampled toward Step 19d's 1,644
    flagged-likely-review negatives) from the trusted human-curated population:
    `label_confidence` in ("human_curated", "registry_confirmed") and `label` in
    ("positive", "negative"). `exclude_ids` removes the validation-fixture record_ids so the
    "easy", hand-picked fixture set can never double-dip into this real evaluation sample.

    Draws each class separately (so the exact requested split is honored regardless of the two
    classes' relative pool sizes), then shuffles the concatenation -- file-ordering hygiene for
    anyone reading an intermediate CSV; the real blinding boundary is `strip_for_api()` below, at
    call time, not row order. Returns the sample WITH its true `label` column intact (needed for
    later kappa scoring) -- this DataFrame must never be passed directly into `prompts.build_prompt`
    without going through `strip_for_api()` first."""
    pool = dataset[
        dataset["label_confidence"].isin(_TRUSTED_LABEL_CONFIDENCE)
        & dataset["label"].isin(["positive", "negative"])
    ]
    if exclude_ids:
        pool = pool[~pool["record_id"].isin(set(exclude_ids))]

    positives = pool[pool["label"] == "positive"]
    negatives = pool[pool["label"] == "negative"]
    if len(positives) < n_positive:
        raise ValueError(f"Only {len(positives)} eligible positives available, need {n_positive}")
    if len(negatives) < n_negative:
        raise ValueError(f"Only {len(negatives)} eligible negatives available, need {n_negative}")

    sampled_positive = positives.sample(n=n_positive, random_state=random_state)
    sampled_negative = negatives.sample(n=n_negative, random_state=random_state)
    combined = pd.concat([sampled_positive, sampled_negative], ignore_index=True)
    return combined.sample(frac=1, random_state=random_state).reset_index(drop=True)


def select_full_population(
    dataset: pd.DataFrame, include_candidates: bool = False, held_out_ids: set | None = None
) -> pd.DataFrame:
    """Step 20e: the population for `classify --scope all` / `--scope all_plus_candidates`.
    `label_confidence` in `_TRUSTED_LABEL_CONFIDENCE`, plus `heuristic_candidate` too when
    `include_candidates=True` (the EPMC "clear negative" background sample -- Step 14's
    `clear_negative_sampler_strong` and Step 19d's `clear_negative_sampler_strong_filtered_v2`,
    fetched via a query designed to exclude AI/ML terms, never individually human-reviewed).
    `held_out_ids` (Step 20's original 1,000-record evaluation sample, `second_curator_sample.csv`)
    is always excluded when passed -- those records already have a real, human-final decision from
    Cross Curate Resolve, so re-classifying them spends money without adding new signal."""
    label_confidences = _TRUSTED_LABEL_CONFIDENCE
    if include_candidates:
        label_confidences = label_confidences + ("heuristic_candidate",)
    pool = dataset[
        dataset["label_confidence"].isin(label_confidences) & dataset["label"].isin(["positive", "negative"])
    ]
    if held_out_ids:
        pool = pool[~pool["record_id"].isin(set(held_out_ids))]
    return pool


def select_bulk_pool_excluding_curated(
    bulk_df: pd.DataFrame, existing_ids: set, record_ids: pd.Series
) -> pd.DataFrame:
    """Step 23a: the full bulk EPMC AI/ML pool (`data/interim/bulk_candidates.csv`, no `label`/
    `label_confidence` columns -- unlike the trusted pool, these are unlabeled candidates), minus
    every record already in the curated set (by pmcid/pmid/doi -- `existing_ids`, the exact same
    `_load_existing_ids(canonical_path)` set every other merge step in this codebase already
    builds, so "already curated" means the identical thing everywhere). `record_ids` is a
    pre-computed Series aligned to `bulk_df`'s index (built by the caller via
    `dedupe.keys.record_id_from_ids` on each row's pmcid/pmid/doi -- the SAME hash
    `canonical_dataset.csv` stamps, so a record found in both pools always gets the same id,
    letting exclusion and any later merge reconcile without a separate join).

    Deliberately pure/no I/O -- `record_ids` and `existing_ids` are both passed in already
    computed, so this function itself never reads a file, matching `select_full_population`'s
    convention above."""
    pool = bulk_df.copy()
    pool["record_id"] = record_ids.values
    id_cols = pool[["pmcid", "pmid", "doi"]].fillna("")
    is_curated = (
        id_cols["pmcid"].isin(existing_ids)
        | id_cols["pmid"].isin(existing_ids)
        | id_cols["doi"].isin(existing_ids)
    )
    return pool[~is_curated].reset_index(drop=True)


def select_staged_file(staged_df: pd.DataFrame) -> pd.DataFrame:
    """Step 24 phase 2: an already-staged batch of genuinely-new records, ready to classify.

    This is the incremental loop's classification population, and it differs from every scope
    above in one way that matters: **the filtering has already happened.**
    `moros_pipeline/scripts/build_incoming_documents.py` fetched only the time windows the coverage
    ledger had never covered, minted each record's deterministic UUID5, and dropped every `_id`
    already present in the corpus. So there is nothing left to exclude here -- re-applying a
    curated-set filter would be wrong (a curated paper reappearing in a new EPMC window is already
    in Mongo and was already dropped) and re-deriving `record_id` would be worse (the staged file's
    `pid` IS the document `_id`, which is what makes the eventual upsert idempotent).

    So this validates and passes through. `pid` becomes `record_id` so the event log, the resume
    logic and the eventual document merge all key on the same identifier -- the mistake this
    avoids is the one `--events-out` was added for: two identifier spaces conflated in one log.

    Records with no abstract are dropped: `strip_for_api` would send a title-only prompt and the
    verdict would be a guess. They stay in the staged CSV and are reported, not silently lost.
    """
    required = {"pid", "title", "abstract", "journal", "year"}
    missing = required - set(staged_df.columns)
    if missing:
        raise ValueError(
            f"staged file is missing {sorted(missing)} -- it must come from "
            f"build_incoming_documents.py, whose output already carries pid and the prompt fields"
        )
    pool = staged_df.copy()
    pool["record_id"] = pool["pid"]
    if pool["record_id"].isna().any() or (pool["record_id"].astype(str).str.strip() == "").any():
        raise ValueError("staged file has rows with no pid -- refusing to classify records that "
                         "cannot be merged back to a document")
    if pool["record_id"].duplicated().any():
        raise ValueError("staged file has duplicate pids -- build_incoming_documents.py "
                         "deduplicates, so this file was not produced by it")
    has_abstract = pool["abstract"].notna() & (pool["abstract"].astype(str).str.strip() != "")
    return pool[has_abstract].reset_index(drop=True)


def strip_for_api(record) -> dict:
    """The actual blinding boundary: returns ONLY {title, abstract, journal, year} from `record`
    (a dict or pandas Series) -- called at the point of building each API request, never earlier,
    so even a bug that accidentally passed a full sample row deeper into the call stack still
    can't leak the label past this one function."""
    return {field: record.get(field) for field in _API_ALLOWED_FIELDS}
