"""Tests for the enrichment step's thinking-effort control and token telemetry.

Why this exists: output tokens are ~94% of this step's cost and ~98% of them are the reasoning
trace (measured on the real Bioinformatics run, 2026-09-03). `reasoning_effort` is the lever, and
the only thing that can go wrong silently is the wire shape -- so the request body is asserted
directly rather than inferred from behaviour.

No paid API call is made anywhere here.
"""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from dome_triage.llm_classify import enrichment as en
from dome_triage.llm_classify.deepseek_client import DeepSeekResponse


# ---------------------------------------------------------------------------
# The wire shape
# ---------------------------------------------------------------------------


def test_no_effort_sends_nothing_so_the_provider_default_stands():
    """Default must be a true no-op: this step's behaviour cannot change until a level is chosen
    from the paired experiment."""
    assert en.thinking_extra_body(None) is None


def test_each_level_maps_to_the_documented_shape():
    assert en.thinking_extra_body("low") == {
        "thinking": {"type": "enabled", "reasoning_effort": "low"}
    }
    assert en.thinking_extra_body("high") == {
        "thinking": {"type": "enabled", "reasoning_effort": "high"}
    }
    assert en.thinking_extra_body("max") == {
        "thinking": {"type": "enabled", "reasoning_effort": "max"}
    }


def test_none_disables_thinking_using_the_confirmed_live_shape():
    """`{"thinking": {"type": "disabled"}}` is the shape deepseek_client.py records as probed live
    (reasoning_tokens -> 0). Not `enable_thinking`, which the same probe found is ignored."""
    assert en.thinking_extra_body("none") == {"thinking": {"type": "disabled"}}


def test_an_unknown_level_is_refused_rather_than_silently_sent():
    # The API would 4xx on this, but failing here names the mistake.
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        en.thinking_extra_body("medium")


# ---------------------------------------------------------------------------
# It reaches the request
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Captures the kwargs `enrich_records` passes, and returns a fixed parseable answer."""

    def __init__(self, reasoning_tokens: int = 900, finish_reason: str = "stop") -> None:
        self.calls: list[dict] = []
        self._reasoning_tokens = reasoning_tokens
        self._finish_reason = finish_reason

    def chat_completion(self, messages, **kwargs):
        self.calls.append(kwargs)
        return DeepSeekResponse(
            content=json.dumps({
                "domain_tier1": [], "domain_tier2": [], "domain_tier3": [],
                "learning_paradigm": [], "model_family": [], "model_type": [],
                "rationale": "r",
            }),
            reasoning_content=None,
            prompt_tokens=3000,
            completion_tokens=1000,
            total_tokens=4000,
            finish_reason=self._finish_reason,
            raw={"usage": {
                "prompt_cache_hit_tokens": 2500,
                "completion_tokens_details": {"reasoning_tokens": self._reasoning_tokens},
            }},
        )


def _records(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": f"r{i}", "title": "T", "abstract": "A", "journal": "J", "year": "2024"}
        for i in range(n)
    ])


_LOOKUP = {
    "domain_tier1": {}, "domain_tier2": {}, "domain_tier3": {},
    "learning_paradigm": {}, "model_family": {}, "model_type_seed": {},
    "_caps": {"domain_tier1": 1, "domain_tier2": 2, "domain_tier3": 3,
              "learning_paradigm": 2, "model_family": 3},
}


def _run(client, **kw) -> list[dict]:
    return list(en.enrich_records(
        _records(), "flash", client, "SYSTEM", _LOOKUP, "batch", max_workers=1, **kw
    ))


def test_the_effort_reaches_chat_completion_as_extra_body():
    client = _RecordingClient()
    _run(client, reasoning_effort="low")
    assert client.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled", "reasoning_effort": "low"}
    }


def test_without_an_effort_extra_body_is_none():
    client = _RecordingClient()
    _run(client)
    assert client.calls[0]["extra_body"] is None


def test_the_max_tokens_cap_is_still_applied():
    client = _RecordingClient()
    _run(client, reasoning_effort="max")
    assert client.calls[0]["max_tokens"] == en.ENRICHMENT_MAX_TOKENS


# ---------------------------------------------------------------------------
# The telemetry
# ---------------------------------------------------------------------------


def test_reasoning_tokens_and_finish_reason_land_on_the_event():
    """Both were captured on the response and discarded until 2026-09-03; both truncation
    incidents on this step had to be diagnosed indirectly from output_tokens == cap."""
    events = _run(_RecordingClient(reasoning_tokens=873, finish_reason="stop"))
    assert events[0]["reasoning_tokens"] == 873
    assert events[0]["finish_reason"] == "stop"


def test_a_truncated_response_is_now_directly_visible():
    events = _run(_RecordingClient(finish_reason="length"))
    assert events[0]["finish_reason"] == "length"


def test_missing_usage_details_do_not_crash_the_run():
    """An older or differently-shaped usage block must degrade to 0, not raise mid-batch."""

    class _NoDetails(_RecordingClient):
        def chat_completion(self, messages, **kwargs):
            response = super().chat_completion(messages, **kwargs)
            response.raw = {"usage": {"prompt_cache_hit_tokens": 0}}
            return response

    assert _run(_NoDetails())[0]["reasoning_tokens"] == 0


def test_both_new_columns_are_in_the_event_schema():
    assert "reasoning_tokens" in en.EVENT_COLUMNS
    assert "finish_reason" in en.EVENT_COLUMNS
    # Every event key must be a declared column, or _stream_classify_events_to_disk drops it.
    assert set(_run(_RecordingClient())[0]) == set(en.EVENT_COLUMNS)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_the_step_and_cli_both_accept_and_forward_the_option():
    assert "reasoning_effort" in inspect.signature(
        __import__("dome_triage.pipeline.steps", fromlist=["x"]).step_llm_classify_enrich
    ).parameters
    from dome_triage import cli

    source = inspect.getsource(cli.llm_classify_enrich)
    assert "--reasoning-effort" in source
    assert "reasoning_effort" in source.split("pipeline_steps.step_llm_classify_enrich")[1]


# ---------------------------------------------------------------------------
# Truncated responses are not retried into an infinite money sink
# ---------------------------------------------------------------------------


def _events(rows: list[dict]) -> pd.DataFrame:
    base = {c: "" for c in en.EVENT_COLUMNS}
    base.update({"model_tier": "flash", "prompt_version": en.ENRICHMENT_PROMPT_VERSION,
                 "vocab_sha256": "HASH", "parse_status": "ok", "finish_reason": "stop"})
    return pd.DataFrame([{**base, **r} for r in rows])


def test_a_truncated_record_is_not_retried_by_default():
    """Measured 2026-09-03: retrying a `finish_reason == "length"` record at the same cap
    re-truncates deterministically -- a resumed run spent ten minutes and ~48,000 output tokens
    re-failing on three records. Retrying it is not free, so it is not automatic."""
    events = _events([{"record_id": "trunc", "parse_status": "parse_error",
                       "finish_reason": "length"}])
    assert "trunc" in en._already_enriched_ids(events, "flash", "HASH")


def test_retry_truncated_forces_it_back_into_the_queue():
    events = _events([{"record_id": "trunc", "parse_status": "parse_error",
                       "finish_reason": "length"}])
    assert "trunc" not in en._already_enriched_ids(events, "flash", "HASH", retry_truncated=True)


def test_an_ordinary_parse_error_is_still_retried_for_free():
    """A malformed-but-complete response can genuinely succeed on a re-ask; only truncation is
    deterministic."""
    events = _events([{"record_id": "bad", "parse_status": "parse_error", "finish_reason": "stop"}])
    assert "bad" not in en._already_enriched_ids(events, "flash", "HASH")


def test_a_successful_record_stays_done():
    events = _events([{"record_id": "good"}])
    assert "good" in en._already_enriched_ids(events, "flash", "HASH")


def test_an_event_log_predating_the_finish_reason_column_still_resumes():
    """The column was added 2026-09-03; older logs (the 3,708-row Step 20j trial) do not have it."""
    events = _events([{"record_id": "old", "parse_status": "parse_error"}]).drop(
        columns=["finish_reason"]
    )
    assert en._already_enriched_ids(events, "flash", "HASH") == set()
