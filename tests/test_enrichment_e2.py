"""Enrichment prompt e2 (GLM-5.3 on Z.ai): frozen at its validated hash, a strict superset of e1 with
e1 untouched, post-processing that logs everything it does, a Z.ai client that sends exactly the
validated request, and a step that resumes and never mixes prompt versions.

No test here makes a network call: the client is driven through a fake session and the step through
a fake client.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest
import requests

from dome_triage.llm_classify import enrichment as e1
from dome_triage.llm_classify import enrichment_e2 as e2
from dome_triage.llm_classify.deepseek_client import DeepSeekResponse
from dome_triage.llm_classify.zai_client import ZaiClient, normalise_usage, projected_usd
from dome_triage.pipeline import steps as pipeline_steps

REPO = Path(__file__).resolve().parents[1]
CRITERIA = REPO / "curation_criteria"
FAKE_KEY = "test-key-not-real"
E1_SHA = "41db952f15118176fa27dac724eff9dff59714b62854859d9606ef440ee473c7"


def render_e1() -> str:
    return e1.build_static_system_text(e1.load_vocabularies(CRITERIA), "flat")


def lookup() -> dict:
    return e2.render_e2(CRITERIA)[2]


# --- frozen prompt, additive vocabulary ---------------------------------------------------------

def test_e2_renders_to_its_validated_hash_and_e1_is_untouched():
    text, _, _ = e2.render_e2(CRITERIA)
    assert e1.vocab_sha256(text) == e2.EXPECTED_E2_SHA256 and e2.e2_problems(text) == []
    assert text == (REPO / "prompts" / "enrichment_system_message.e2.txt").read_text(encoding="utf-8")
    assert e1.vocab_sha256(render_e1()) == E1_SHA
    hashes = json.loads((REPO / "prompts" / "PROMPT_HASHES.json").read_text())
    assert hashes["enrichment"]["vocab_sha256"] == E1_SHA
    assert hashes["enrichment_e2"]["vocab_sha256"] == e2.EXPECTED_E2_SHA256


def test_e2_problems_refuses_a_drifted_render():
    text, _, _ = e2.render_e2(CRITERIA)
    assert e2.e2_problems(text + " ")


def test_e2_vocabulary_is_a_strict_superset_of_e1_with_provenance():
    e1_branch = json.loads((CRITERIA / "modelling_branch_vocab.json").read_text())["fields"]
    e1_seed = json.loads((CRITERIA / "model_type_seed_vocab.json").read_text())["terms"]
    branch = json.loads((CRITERIA / "e2" / "modelling_branch_vocab.json").read_text())["fields"]
    seed = json.loads((CRITERIA / "e2" / "model_type_seed_vocab.json").read_text())["terms"]
    for field in ("learning_paradigm", "model_family"):
        old = {t["label"] for t in e1_branch[field]["terms"]}
        new = {t["label"]: t for t in branch[field]["terms"]}
        assert old <= set(new) and branch[field]["max_tags"] == e1_branch[field]["max_tags"]
        for label in set(new) - old:
            assert new[label]["extension"] == "e2"
            assert new[label]["provenance"].startswith("curation_criteria/e2/PROVENANCE.md#e2-v")
    assert {t["label"] for t in branch["model_use"]["terms"]} == {
        "trained in this study", "fine-tuned or adapted", "pretrained, applied as-is", "prompted (zero/few-shot)"}
    by = {t["canonical"]: t for t in seed}
    for t in e1_seed:
        assert set(t["aliases"]) <= set(by[t["canonical"]]["aliases"])
        assert by[t["canonical"]]["family_hint"] == t["family_hint"]
    provenance = (CRITERIA / "e2" / "PROVENANCE.md").read_text()
    for change in [f"E2-V{i}" for i in range(1, 10)] + [f"E2-P{i}" for i in range(1, 7)] + ["E2-PP1", "E2-PP2", "E2-PP3"]:
        assert change in provenance
    assert "## Validation" in provenance


def test_user_message_is_e1s_and_blinded():
    record = {"record_id": "x", "title": "T", "abstract": "A", "journal": "J", "year": "2024", "label": "secret"}
    e2_text, _, _ = e2.render_e2(CRITERIA)
    assert e1.build_enrichment_prompt(record, render_e1())[1] == e1.build_enrichment_prompt(record, e2_text)[1]
    assert "secret" not in json.dumps(e1.build_enrichment_prompt(record, e2_text))


# --- parsing: logged post-processing ---------------------------------------------------------------

def answer(**fields):
    base = {"domain_tier1": [], "domain_tier2": [], "domain_tier3": [], "model_use": ["trained in this study"],
            "learning_paradigm": ["supervised"], "model_family": [], "model_type": [], "rationale": "r"}
    return json.dumps({**base, **fields})


def test_family_backfill_is_logged_and_respects_the_cap():
    out = e2.parse_e2(answer(model_family=["classical machine learning"], model_type=["RF", "PCA"]), lookup())
    assert out["model_type"] == ["random forest", "principal component analysis"]
    assert out["model_family"] == ["classical machine learning", "ensemble learning", "dimensionality reduction"]
    assert out["family_backfilled"] == ["ensemble learning<-random forest",
                                        "dimensionality reduction<-principal component analysis"]
    full = e2.parse_e2(answer(model_family=["deep learning", "transfer learning", "classical machine learning"],
                              model_type=["XGBoost"]), lookup())
    assert full["model_family"] == ["deep learning", "transfer learning", "classical machine learning"]
    assert full["family_backfilled"] == ["skipped_cap:ensemble learning<-XGBoost"]


def test_ambiguous_and_non_family_hints_are_never_backfilled():
    out = e2.parse_e2(answer(model_type=["ANN", "PPO"]), lookup())
    assert out["model_type"] == ["neural network", "policy gradient"]
    assert out["model_family"] == [] and out["family_backfilled"] == []


def test_paradigm_repair_and_model_use_are_logged():
    out = e2.parse_e2(answer(learning_paradigm=["reinforcement learning"], model_use=["trained here"]), lookup())
    assert out["learning_paradigm"] == ["reinforcement"]
    assert "learning_paradigm:repaired:reinforcement learning->reinforcement" in out["vocab_violations"]
    assert out["model_use"] == ["trained here"] and "model_use:unknown:trained here" in out["vocab_violations"]
    ok = e2.parse_e2(answer(learning_paradigm=["self-supervised", "supervised"],
                            model_use=["prompted (zero/few-shot)"]), lookup())
    assert ok["vocab_violations"] == [] and ok["model_use"] == ["prompted (zero/few-shot)"]


def test_excluded_kinds_are_flagged_not_dropped():
    out = e2.parse_e2(answer(model_type=["SHAP", "WGCNA", "molecular docking", "deep learning", "random forest",
                                         "particle swarm optimization"]), lookup())
    assert out["model_type"][0] == "SHAP" and len(out["model_type"]) == 6
    kinds = sorted(v.split(":")[2] for v in out["vocab_violations"] if v.startswith("model_type:excluded_kind"))
    assert kinds == ["explainability", "family_name", "optimiser_check", "pipeline", "simulation"]


def test_parse_error_carries_empty_e2_fields():
    out = e2.parse_e2("not json", lookup())
    assert out["parse_status"] == e1.PARSE_ERROR and out["model_use"] == [] and out["family_backfilled"] == []


# --- Z.ai client ------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response

    def close(self):
        pass


def ok_payload():
    return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6000, "completion_tokens": 400, "total_tokens": 6400,
                      "prompt_tokens_details": {"cached_tokens": 2500}}}


def test_zai_request_is_exactly_the_validated_one():
    session = FakeSession(FakeResponse(payload=ok_payload()))
    client = ZaiClient(FAKE_KEY, session=session)
    text, _, _ = e2.render_e2(CRITERIA)
    messages = e1.build_enrichment_prompt({"title": "T", "abstract": "A", "journal": "J", "year": "2024"}, text)
    response = client.chat_completion(messages, tier="glm-5.3-flash", max_tokens=e1.ENRICHMENT_MAX_TOKENS)
    call = session.calls[0]
    body = call["json"]
    assert call["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert body["model"] == "glm-5.3-flash" and body["messages"] == messages
    assert body["max_tokens"] == 16000 and body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert "tools" not in body and call["timeout"] == 300.0
    assert response.raw["usage"]["prompt_cache_hit_tokens"] == 2500


def test_zai_client_refuses_what_the_validated_config_never_had():
    client = ZaiClient(FAKE_KEY, session=FakeSession(FakeResponse(payload=ok_payload())))
    with pytest.raises(ValueError):
        client.chat_completion([], tier="glm-5.3-flash", enable_search=True)
    with pytest.raises(ValueError):
        client.chat_completion([], tier="glm-5.3-flash", extra_body={"thinking": {"type": "disabled"}})
    with pytest.raises(ValueError):
        client.chat_completion([], tier="flash")
    with pytest.raises(ValueError):
        ZaiClient("")
    with pytest.raises(ValueError):
        ZaiClient(FAKE_KEY, reasoning_effort="extreme")
    assert FAKE_KEY not in repr(client)


def test_zai_http_error_does_not_echo_the_key():
    client = ZaiClient(FAKE_KEY, session=FakeSession(FakeResponse(status=400, text='{"error":"bad"}')))
    with pytest.raises(requests.HTTPError) as exc:
        client.chat_completion([], tier="glm-5.3-flash")
    assert FAKE_KEY not in str(exc.value) and "400" in str(exc.value)


def test_normalise_usage_and_projection():
    assert normalise_usage({"usage": {"prompt_tokens": 10}})["usage"]["prompt_cache_hit_tokens"] == 0
    # The measured wave2 profile reproduces that run's list price ($0.2398 for 250) at headroom 1.
    assert projected_usd("glm-5.3-flash", 250, headroom=1.0) == pytest.approx(0.2398, abs=0.001)


# --- the step ---------------------------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()

    def chat_completion(self, messages, tier, max_tokens=6000, extra_body=None, **kwargs):
        with self._lock:
            self.messages.append(messages)
        return DeepSeekResponse(content=answer(model_type=["XGBoost"]), reasoning_content=None, prompt_tokens=6000,
                                completion_tokens=500, total_tokens=6500, finish_reason="stop",
                                raw={"usage": {"prompt_cache_hit_tokens": 5000,
                                               "completion_tokens_details": {"reasoning_tokens": 300}}})

    def close(self):
        pass


def _input(tmp_path) -> Path:
    path = tmp_path / "input.csv"
    pd.DataFrame([{"record_id": "a", "title": "T", "abstract": "XGBoost.", "journal": "J", "year": "2024"},
                  {"record_id": "b", "title": "T2", "abstract": "SVM.", "journal": "J", "year": "2025"}]
                 ).to_csv(path, index=False)
    return path


def test_step_writes_e2_events_and_resumes(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(pipeline_steps, "_zai_client", lambda *a, **k: client)
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)
    events = tmp_path / "events.csv"
    pipeline_steps.step_llm_classify_enrich_e2(None, input_path=_input(tmp_path), events_out=str(events), concurrency=2)
    df = pd.read_csv(events, dtype=str)
    assert list(df.columns) == e2.E2_EVENT_COLUMNS and len(df) == 2
    assert set(df["prompt_version"]) == {"e2"} and set(df["vocab_sha256"]) == {e2.EXPECTED_E2_SHA256}
    assert set(df["model_tier"]) == {"glm-5.3-flash"}
    assert json.loads(df.set_index("record_id").loc["a", "model_family"]) == ["classical machine learning",
                                                                              "ensemble learning"]
    assert json.loads(df.loc[0, "model_use"]) == ["trained in this study"]
    assert client.messages[0][0]["content"] == (REPO / "prompts" / "enrichment_system_message.e2.txt").read_text()

    again = FakeClient()
    monkeypatch.setattr(pipeline_steps, "_zai_client", lambda *a, **k: again)
    pipeline_steps.step_llm_classify_enrich_e2(None, input_path=_input(tmp_path), events_out=str(events), concurrency=2)
    assert again.messages == [] and len(pd.read_csv(events)) == 2


def test_step_refuses_deepseek_tiers_mixed_logs_and_missing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_steps, "_zai_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)
    src = _input(tmp_path)
    with pytest.raises(ValueError, match="Z.ai tiers"):
        pipeline_steps.step_llm_classify_enrich_e2(None, tier="flash", input_path=src, events_out=str(tmp_path / "x.csv"))
    with pytest.raises(ValueError, match="--input"):
        pipeline_steps.step_llm_classify_enrich_e2(None, events_out=str(tmp_path / "x.csv"))
    with pytest.raises(ValueError, match="--events-out"):
        pipeline_steps.step_llm_classify_enrich_e2(None, input_path=src)
    e1_log = tmp_path / "e1_events.csv"
    row = {c: "" for c in e1.EVENT_COLUMNS}
    row.update(record_id="a", prompt_version="e1", parse_status="ok", model_tier="flash")
    pd.DataFrame([row]).to_csv(e1_log, index=False)
    with pytest.raises(ValueError, match="another prompt version"):
        pipeline_steps.step_llm_classify_enrich_e2(None, input_path=src, events_out=str(e1_log))


def test_step_needs_the_zai_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        pipeline_steps._zai_client(10, None)


def test_cli_routes_e2_and_keeps_e1_defaults():
    import inspect

    from dome_triage import cli

    source = inspect.getsource(cli.llm_classify_enrich)
    assert "step_llm_classify_enrich_e2" in source
    params = inspect.signature(cli.llm_classify_enrich).parameters
    assert params["tier"].default.default == "flash" and params["concurrency"].default.default == 800
    assert params["prompt"].default.default == "e1"
