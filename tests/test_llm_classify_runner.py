import pandas as pd

from dome_triage.llm_classify.deepseek_client import DeepSeekResponse
from dome_triage.llm_classify.prompts import PROMPT_VERSION, criteria_sha256
from dome_triage.llm_classify.runner import (
    EVENT_COLUMNS,
    classify_records,
    run_criteria_validation,
    select_undetermined_subset,
)

_CRITERIA_TEXT = "## Positive\nApplies AI/ML.\n## Negative\nDoes not."
_CRITERIA_HASH = criteria_sha256(_CRITERIA_TEXT)


class _FakeClient:
    def __init__(self, canned_content: str = '{"classification": "positive", "rationale": "x"}'):
        self.canned_content = canned_content
        self.calls: list[dict] = []

    def chat_completion(self, messages, tier, max_tokens=400, temperature=0.0, request_json_mode=True, enable_search=False):
        self.calls.append({"messages": messages, "tier": tier, "enable_search": enable_search})
        return DeepSeekResponse(
            content=self.canned_content,
            reasoning_content=None,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )


class _FailingClient:
    """Raises for one specific record_id, succeeds for everything else -- used to test that a
    single failed call under concurrency doesn't lose the other in-flight results."""

    def __init__(self, fail_on_title: str):
        self.fail_on_title = fail_on_title
        self.calls: list[dict] = []

    def chat_completion(self, messages, tier, max_tokens=400, temperature=0.0, request_json_mode=True, enable_search=False):
        self.calls.append({"messages": messages, "tier": tier})
        if self.fail_on_title in messages[1]["content"]:
            raise RuntimeError("simulated network failure")
        return DeepSeekResponse(
            content='{"classification": "positive", "rationale": "x"}',
            reasoning_content=None,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"record_id": "r1", "title": "T1", "abstract": "A1", "journal": "J", "year": "2020", "label": "positive"},
            {"record_id": "r2", "title": "T2", "abstract": "A2", "journal": "J", "year": "2020", "label": "negative"},
        ]
    )


def test_classify_records_yields_one_event_per_record():
    client = _FakeClient()
    events = list(
        classify_records(_sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
                          criteria_hash=_CRITERIA_HASH, batch_id="batch1")
    )
    assert len(events) == 2
    assert {e["record_id"] for e in events} == {"r1", "r2"}
    assert all(e["classification"] == "positive" for e in events)
    assert all(e["mode"] == "primary" for e in events)
    assert all(e["prompt_version"] == PROMPT_VERSION for e in events)


def test_classify_records_skips_already_classified_ids():
    client = _FakeClient()
    existing_events = pd.DataFrame(
        [
            {
                "record_id": "r1", "classification": "positive", "rationale": "", "model_tier": "flash",
                "mode": "primary", "prompt_version": PROMPT_VERSION, "criteria_sha256": _CRITERIA_HASH,
                "batch_id": "old", "input_tokens": 1, "output_tokens": 1, "parse_fallback_used": "False",
                "used_search": "False", "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
        columns=EVENT_COLUMNS,
    )
    events = list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch2", existing_events=existing_events,
        )
    )
    assert len(events) == 1
    assert events[0]["record_id"] == "r2"
    assert len(client.calls) == 1


def test_classify_records_retries_a_prior_parse_error_instead_of_skipping_it():
    """A parse_error row is a genuine failure, not a completed result -- regression test for a
    real incident: 84/1000 real paid calls came back as parse_error, and the resumability check
    originally treated them as "already classified" too, meaning simply re-running the command
    would have silently skipped every broken row forever instead of retrying it."""
    client = _FakeClient()
    existing_events = pd.DataFrame(
        [
            {
                "record_id": "r1", "classification": "parse_error", "rationale": "truncated", "model_tier": "flash",
                "mode": "primary", "prompt_version": PROMPT_VERSION, "criteria_sha256": _CRITERIA_HASH,
                "batch_id": "old", "input_tokens": 1, "output_tokens": 1, "parse_fallback_used": "True",
                "used_search": "False", "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
        columns=EVENT_COLUMNS,
    )
    events = list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch2", existing_events=existing_events,
        )
    )
    assert {e["record_id"] for e in events} == {"r1", "r2"}  # r1 retried, not skipped


def test_classify_records_does_not_skip_a_different_criteria_hash():
    """Resumability keys on criteria_sha256 -- a CRITERIA.md edit must never be silently conflated
    with an earlier run's results."""
    client = _FakeClient()
    existing_events = pd.DataFrame(
        [
            {
                "record_id": "r1", "classification": "positive", "rationale": "", "model_tier": "flash",
                "mode": "primary", "prompt_version": PROMPT_VERSION, "criteria_sha256": "a-different-hash",
                "batch_id": "old", "input_tokens": 1, "output_tokens": 1, "parse_fallback_used": "False",
                "used_search": "False", "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
        columns=EVENT_COLUMNS,
    )
    events = list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch2", existing_events=existing_events,
        )
    )
    assert len(events) == 2


def test_classify_records_forced_guess_mode_uses_forced_choice_prompt():
    client = _FakeClient()
    list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch1", mode="forced_guess",
        )
    )
    system_content = client.calls[0]["messages"][0]["content"]
    assert "never allowed to answer" in system_content.lower()


def test_classify_records_rag_mode_enables_search_on_the_client_call():
    client = _FakeClient()
    list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch1", mode="rag", enable_search=True,
        )
    )
    assert all(call["enable_search"] is True for call in client.calls)


def test_classify_records_primary_mode_never_enables_search_even_if_flag_omitted():
    client = _FakeClient()
    list(
        classify_records(
            _sample_df(), tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch1",
        )
    )
    assert all(call["enable_search"] is False for call in client.calls)


def test_classify_records_runs_concurrently_and_yields_all_results():
    client = _FakeClient()
    sample_df = pd.DataFrame(
        [
            {"record_id": f"r{i}", "title": f"T{i}", "abstract": f"A{i}", "journal": "J", "year": "2020", "label": "positive"}
            for i in range(12)
        ]
    )
    events = list(
        classify_records(
            sample_df, tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch1", max_workers=5,
        )
    )
    assert {e["record_id"] for e in events} == {f"r{i}" for i in range(12)}
    assert len(client.calls) == 12


def test_classify_records_one_failure_does_not_lose_other_in_flight_results():
    """A single bad call under concurrency must not cost the results of every other call already
    dispatched alongside it -- this is the exact failure mode a real 840-call run hit."""
    client = _FailingClient(fail_on_title="T3")
    sample_df = pd.DataFrame(
        [
            {"record_id": f"r{i}", "title": f"T{i}", "abstract": f"A{i}", "journal": "J", "year": "2020", "label": "positive"}
            for i in range(8)
        ]
    )
    events = []
    raised = None
    try:
        for event in classify_records(
            sample_df, tier="flash", client=client, criteria_text=_CRITERIA_TEXT,
            criteria_hash=_CRITERIA_HASH, batch_id="batch1", max_workers=4,
        ):
            events.append(event)
    except RuntimeError as exc:
        raised = exc

    assert raised is not None
    assert "simulated network failure" in str(raised)
    # The other 7 records (everything except r3/"T3") must still have been yielded -- not lost.
    assert {e["record_id"] for e in events} == {f"r{i}" for i in range(8) if i != 3}


def test_select_undetermined_subset_returns_only_undetermined_records():
    sample_df = _sample_df()
    primary_events = pd.DataFrame(
        [
            {
                "record_id": "r1", "classification": "undeterminable", "rationale": "", "model_tier": "flash",
                "mode": "primary", "prompt_version": PROMPT_VERSION, "criteria_sha256": _CRITERIA_HASH,
                "batch_id": "b", "input_tokens": 1, "output_tokens": 1, "parse_fallback_used": "False",
                "used_search": "False", "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "record_id": "r2", "classification": "negative", "rationale": "", "model_tier": "flash",
                "mode": "primary", "prompt_version": PROMPT_VERSION, "criteria_sha256": _CRITERIA_HASH,
                "batch_id": "b", "input_tokens": 1, "output_tokens": 1, "parse_fallback_used": "False",
                "used_search": "False", "timestamp": "2026-01-01T00:00:00+00:00",
            },
        ],
        columns=EVENT_COLUMNS,
    )
    subset = select_undetermined_subset(sample_df, primary_events, tier="flash", criteria_hash=_CRITERIA_HASH)
    assert list(subset["record_id"]) == ["r1"]


def test_select_undetermined_subset_empty_when_no_primary_events():
    sample_df = _sample_df()
    subset = select_undetermined_subset(sample_df, pd.DataFrame(columns=EVENT_COLUMNS), tier="flash", criteria_hash=_CRITERIA_HASH)
    assert subset.empty


def test_run_criteria_validation_runs_both_variants_and_scores_match():
    client = _FakeClient(canned_content='{"classification": "positive", "rationale": "clear case"}')
    fixtures = pd.DataFrame(
        [
            {
                "record_id": "f1", "title": "T", "abstract": "A", "journal": "J", "year": "2020",
                "expected_classification": "positive", "criterion_tested": "classical_ml",
            }
        ]
    )
    result = run_criteria_validation(fixtures, tier="flash", client=client, criteria_text=_CRITERIA_TEXT)
    assert set(result["variant"]) == {"primary", "forced_choice"}
    assert result["match"].all()
    assert len(client.calls) == 2
