"""The actual paid classify loop(s) for Step 20 -- resumable (skips any record already logged for
the same tier/mode/prompt_version/criteria_sha256 combination so a crash mid-run never requires
re-paying), never touches canonical_dataset.csv -- this is a comparison signal only, per
AGENTS.md's "human curation is never bypassed" rule.

`classify_records` runs up to `max_workers` API calls concurrently via a thread pool (DeepSeek's
own confirmed pricing page states concurrency limits of 500-2500, so single-call-at-a-time was
needlessly slow -- a real, confirmed-live 1000-record run took ~40+ minutes serially). Despite the
concurrency, results are still yielded one at a time to the caller's single consuming thread (see
`pipeline/steps.py::_stream_classify_events_to_disk`) -- no locking needed for the disk write
itself, since only that one thread ever calls it; the concurrency is entirely internal to this
function's network-call dispatch. If one call ultimately fails (after its own internal retries in
`deepseek_client.chat_completion` are exhausted), every other already-in-flight call is still
allowed to finish and still gets yielded/saved -- not cancelled -- so a single bad call loses at
most itself; the first such failure is re-raised only after every dispatched call has drained.

Three call shapes share this module:
1. `run_criteria_validation` -- the cheap, first, "is the prompt actually working" check over the
   hand-picked fixture set, run for BOTH prompt variants so the primary-variant decision (3-way vs
   forced-choice) is made from real numbers. Kept serial -- only ~28 calls, speed doesn't matter.
2. `classify_records` -- the primary 500+500 paid run.
3. `run_forced_guess_fallback` / `run_rag_fallback` -- small-batch re-asks over the primary run's
   undetermined subset (see STEPS_Progress.md Step 20, "Undeterminable rate and RAG fallback").
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from dome_triage.llm_classify.deepseek_client import DeepSeekClient
from dome_triage.llm_classify.prompts import PROMPT_VERSION, build_forced_choice_prompt, build_prompt
from dome_triage.llm_classify.response_parser import PARSE_ERROR, parse_classification
from dome_triage.llm_classify.sampling import strip_for_api

EVENT_COLUMNS = [
    "record_id",
    "classification",
    "rationale",
    "model_tier",
    "mode",
    "prompt_version",
    "criteria_sha256",
    "batch_id",
    "input_tokens",
    "output_tokens",
    "parse_fallback_used",
    "used_search",
    "timestamp",
]

VALIDATION_COLUMNS = [
    "record_id",
    "variant",
    "criterion_tested",
    "expected",
    "actual",
    "match",
    "rationale",
]

_MODE_BUILDERS = {
    "primary": build_prompt,
    "forced_guess": build_forced_choice_prompt,
    "rag": build_prompt,  # same 3-way prompt as primary; enable_search=True is what differs.
}


def run_criteria_validation(
    fixtures: pd.DataFrame, tier: str, client: DeepSeekClient, criteria_text: str
) -> pd.DataFrame:
    """Runs BOTH the primary (3-way) and forced-choice (2-way) prompt variants over the hand-picked
    validation fixtures (`curation_criteria/validation_fixtures.csv`) -- flash tier only in
    practice, since the same prompt drives both DeepSeek tiers by construction, so there's no
    reason to spend at pro-tier prices before the prompt itself is proven correct. Returns one row
    per (record_id, variant) with the model's answer, whether it matched the fixture's own
    `expected_classification`, and the rationale. The CLI layer uses this to print a pass/fail
    table AND the primary variant's undetermined rate -- the number that decides which variant
    drives the real 500+500 sample (see STEPS_Progress.md Step 20's decision rule)."""
    rows = []
    for variant, builder in (("primary", build_prompt), ("forced_choice", build_forced_choice_prompt)):
        for _, fixture in fixtures.iterrows():
            record = {
                "title": fixture.get("title"),
                "abstract": fixture.get("abstract"),
                "journal": fixture.get("journal"),
                "year": fixture.get("year"),
            }
            response = client.chat_completion(builder(record, criteria_text), tier=tier)
            parsed = parse_classification(response.content)
            rows.append(
                {
                    "record_id": fixture["record_id"],
                    "variant": variant,
                    "criterion_tested": fixture.get("criterion_tested"),
                    "expected": fixture["expected_classification"],
                    "actual": parsed.classification,
                    "match": parsed.classification == fixture["expected_classification"],
                    "rationale": parsed.rationale,
                }
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def _already_classified_ids(events: pd.DataFrame, tier: str, mode: str, criteria_hash: str) -> set:
    """A `parse_error` row does NOT count as "already classified" -- it's a genuine failure to get
    a usable answer (real, confirmed-live incident: 84/1000 real paid calls came back truncated,
    see deepseek_client.py's chat_completion docstring), not a completed result. Excluding it here
    is the actual "repair" mechanism for a parse_error row: simply re-running the exact same
    classify command retries every parse_error record automatically (since it's no longer treated
    as done) while still skipping every record that already has a real positive/negative/
    undeterminable answer -- no separate repair command needed."""
    if events.empty:
        return set()
    matches = events[
        (events["model_tier"] == tier)
        & (events["mode"] == mode)
        & (events["prompt_version"] == PROMPT_VERSION)
        & (events["criteria_sha256"] == criteria_hash)
        & (events["classification"] != PARSE_ERROR)
    ]
    return set(matches["record_id"])


def classify_records(
    sample_df: pd.DataFrame,
    tier: str,
    client: DeepSeekClient,
    criteria_text: str,
    criteria_hash: str,
    batch_id: str,
    mode: str = "primary",
    existing_events: Optional[pd.DataFrame] = None,
    enable_search: bool = False,
    max_workers: int = 20,
) -> Iterator[dict]:
    """`mode` is "primary" (3-way, closed-book), "forced_guess" (2-way, closed-book -- the
    forced-guess fallback), or "rag" (3-way, `enable_search=True` -- the one deliberate, narrowly-
    scoped exception to "no tools key ever" anywhere in this design). Yields one event dict per
    newly-classified record, as soon as it completes (NOT in `sample_df`'s row order, since up to
    `max_workers` calls race concurrently -- callers must never assume yield order); records
    already present in `existing_events` for this exact (tier, mode, prompt_version,
    criteria_sha256) combination are skipped, not re-paid for -- a crash mid-run never requires
    re-paying for already-classified records.

    `max_workers` (default 20) runs that many API calls concurrently via a thread pool --
    comfortably under DeepSeek's own documented concurrency limits (500 for pro, 2500 for flash,
    per the confirmed pricing page), while still leaving headroom rather than maxing them out.
    `requests.Session` is safe for concurrent use across threads (the standard, documented
    pattern); `deepseek_client.create_session`'s connection pool is sized to match. If one call
    ultimately fails, every other already-dispatched call is still allowed to finish (not
    cancelled) and its result is still yielded -- the first failure is re-raised only after every
    dispatched call has drained, so a single bad call loses at most itself."""
    existing_events = existing_events if existing_events is not None else pd.DataFrame(columns=EVENT_COLUMNS)
    done_ids = _already_classified_ids(existing_events, tier, mode, criteria_hash)
    builder = _MODE_BUILDERS[mode]

    to_classify = sample_df[~sample_df["record_id"].isin(done_ids)]
    records = [record for _, record in to_classify.iterrows()]

    def _classify_one(record) -> dict:
        prompt = builder(strip_for_api(record), criteria_text)
        response = client.chat_completion(prompt, tier=tier, enable_search=enable_search)
        parsed = parse_classification(response.content)
        return {
            "record_id": record["record_id"],
            "classification": parsed.classification,
            "rationale": parsed.rationale,
            "model_tier": tier,
            "mode": mode,
            "prompt_version": PROMPT_VERSION,
            "criteria_sha256": criteria_hash,
            "batch_id": batch_id,
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.completion_tokens,
            "parse_fallback_used": str(parsed.parse_fallback_used),
            "used_search": str(enable_search),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    first_exception: Optional[BaseException] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_classify_one, record) for record in records]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc=f"llm-classify:{tier}:{mode}"
        ):
            try:
                yield future.result()
            except Exception as exc:
                if first_exception is None:
                    first_exception = exc

    if first_exception is not None:
        raise first_exception


def select_undetermined_subset(
    sample_df: pd.DataFrame, primary_events: pd.DataFrame, tier: str, criteria_hash: str
) -> pd.DataFrame:
    """Records from `sample_df` whose latest PRIMARY-mode classification (this exact tier +
    criteria_sha256) came back "undeterminable" -- the population both fallback batches (§
    Undeterminable rate and RAG fallback) run over. Empty if the primary run hasn't classified
    anything as undeterminable yet (or hasn't run at all)."""
    if primary_events.empty:
        return sample_df.iloc[0:0]
    primary = primary_events[
        (primary_events["model_tier"] == tier)
        & (primary_events["mode"] == "primary")
        & (primary_events["criteria_sha256"] == criteria_hash)
    ]
    if primary.empty:
        return sample_df.iloc[0:0]
    latest = primary.sort_values("timestamp").groupby("record_id").last()
    undetermined_ids = set(latest[latest["classification"] == "undeterminable"].index)
    return sample_df[sample_df["record_id"].isin(undetermined_ids)]
