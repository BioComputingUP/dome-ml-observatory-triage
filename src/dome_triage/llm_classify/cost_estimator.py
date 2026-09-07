"""Real-calibration cost estimation for Step 20 -- deliberately contains NO hardcoded $/token
constant anywhere in this module. Three separate pricing lookups this session (a general web
search, a site-restricted web search, and a fetch of a claimed official pricing page) returned
mutually inconsistent numbers and even disagreed on current DeepSeek model names, so cost is
derived from real, small, live API calls plus the real dollar amount read off Gavin's own DeepSeek
dashboard balance -- never trusted from any looked-up figure.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from dome_triage.llm_classify.deepseek_client import DeepSeekClient
from dome_triage.llm_classify.prompts import build_forced_choice_prompt, build_prompt
from dome_triage.llm_classify.sampling import strip_for_api

_MODE_BUILDERS = {
    "primary": build_prompt,
    "forced_guess": build_forced_choice_prompt,
    "rag": build_prompt,  # same 3-way prompt as primary; enable_search=True is what differs.
}

CALIBRATION_LOG_COLUMNS = [
    "batch_id",
    "tier",
    "mode",
    "record_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "timestamp",
]


def run_calibration_batch(
    records: pd.DataFrame,
    tier: str,
    client: DeepSeekClient,
    criteria_text: str,
    batch_id: str,
    n: int = 8,
    mode: str = "primary",
    enable_search: bool = False,
) -> pd.DataFrame:
    """Fires `n` REAL, live API calls against `tier`, through the exact same prompt-building path
    as a real classify call (`prompts.build_prompt` -- token counts must reflect what a real call
    actually costs, not a simplified stand-in). `mode` selects the same prompt builder
    `runner.classify_records` would use for that mode ("primary"/"forced_guess" -&gt; the matching
    prompt; "rag" -&gt; the primary 3-way prompt, since `enable_search` is what differs there, not the
    prompt text). Returns one row per call with the real token usage read directly from each
    response's `usage` field. Does not write to disk itself -- the caller (`pipeline/steps.py`)
    appends the result to `second_curator_calibration_log.csv`."""
    builder = _MODE_BUILDERS.get(mode, build_prompt)
    sample = records.head(n)
    rows = []
    for _, record in sample.iterrows():
        prompt = builder(strip_for_api(record), criteria_text)
        response = client.chat_completion(prompt, tier=tier, enable_search=enable_search)
        rows.append(
            {
                "batch_id": batch_id,
                "tier": tier,
                "mode": mode,
                "record_id": record["record_id"],
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_LOG_COLUMNS)


def project_cost(
    calibration_log: pd.DataFrame,
    tier: str,
    observed_usd_spent: float,
    target_n: int,
    mode: str = "primary",
    batch_id: str | None = None,
) -> dict:
    """The only trustworthy cost figure anywhere in this design: empirical
    `usd_per_record = observed_usd_spent / (number of calibration rows in ONE exact batch)`, where
    `observed_usd_spent` is the real dollar amount read off Gavin's own DeepSeek dashboard balance
    delta for that exact calibration batch, passed in as a CLI flag -- never looked up or assumed.
    Linearly extrapolates to `target_n`.

    **Real bug fixed here (2026-08-21)**: this used to divide `observed_usd_spent` by every row
    ever logged for this tier+mode, across every calibration run in this dataset's history -- not
    just the one batch the dashboard reading was actually for. Harmless the first time this was
    ever run (only one batch existed), but silently wrong the moment a second calibration batch for
    the same tier+mode existed (e.g. a fresh calibration after a `CRITERIA.md` edit) -- the real
    dashboard $ for the NEW, longer-prompt batch would get diluted across the OLD, shorter-prompt
    batch's row count too, understating the true cost. Fixed by scoping to exactly one `batch_id`:
    the caller's explicit `batch_id` if given, otherwise the most recent batch present for this
    tier+mode (by max timestamp) -- matching the "exact calibration batch" the docstring always
    claimed this was doing.

    Raises `ValueError` if there's no calibration data for this tier+mode yet (nothing to divide
    by) -- never silently returns a zero/None cost."""
    subset = calibration_log[(calibration_log["tier"] == tier) & (calibration_log["mode"] == mode)]
    if subset.empty:
        raise ValueError(
            f"No calibration rows found for tier={tier!r} mode={mode!r} -- run `llm-classify "
            "calibrate` (or `calibrate-fallback`) for this tier/mode first."
        )
    resolved_batch_id = batch_id or subset.sort_values("timestamp")["batch_id"].iloc[-1]
    subset = subset[subset["batch_id"] == resolved_batch_id]
    n_calls = len(subset)
    usd_per_record = observed_usd_spent / n_calls
    return {
        "tier": tier,
        "mode": mode,
        "batch_id": resolved_batch_id,
        "n_calibration_calls": n_calls,
        "observed_usd_spent": observed_usd_spent,
        "usd_per_record": usd_per_record,
        "avg_prompt_tokens": float(pd.to_numeric(subset["prompt_tokens"]).mean()),
        "avg_completion_tokens": float(pd.to_numeric(subset["completion_tokens"]).mean()),
        "projected_usd_for_target_n": usd_per_record * target_n,
        "target_n": target_n,
    }
