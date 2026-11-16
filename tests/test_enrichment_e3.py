"""Enrichment prompt e3 (the latest, production prompt) and the EMBL-EBI provider.

What must hold:
- e3 is frozen at its pinned hash, and differs from e2 only in the model-type seed lines: every other
  block of the prompt is byte-identical, every e2 seed term and alias survives, and the modelling-branch
  vocabulary file is e2's byte for byte. e1 and e2 still render to their own hashes.
- The new aliases normalise to their canonical spellings.
- The EBI client sends what the gateway honours (vLLM's enable_thinking; no ignored effort knob), takes
  its address, key and model only from the environment, and never echoes the key.
- The step writes e3 events with the served model_id, resumes, and is what `enrich` runs by default.

No test makes a network call.
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
from dome_triage.llm_classify import enrichment_e3 as e3
from dome_triage.llm_classify import zai_client
from dome_triage.llm_classify.deepseek_client import DeepSeekResponse
from dome_triage.llm_classify.zai_client import EbiClient
from dome_triage.pipeline import steps as pipeline_steps

REPO = Path(__file__).resolve().parents[1]
CRITERIA = REPO / "curation_criteria"
FAKE_KEY = "test-key-not-real"
SERVED = "zai-org/GLM-5.3-Flash"


def render_e2() -> str:
    return e2.render_e2(CRITERIA)[0]


def render_e3() -> str:
    return e3.render_e3(CRITERIA)[0]


def seed(version: str) -> dict:
    return json.loads((CRITERIA / version / "model_type_seed_vocab.json").read_text(encoding="utf-8"))


# --- frozen prompt, additive vocabulary -----------------------------------------------------------

def test_e3_renders_to_its_pinned_hash_and_matches_the_committed_file():
    text = render_e3()
    assert e1.vocab_sha256(text) == e3.EXPECTED_E3_SHA256 and e3.e3_problems(text) == []
    assert text == (REPO / "prompts" / "enrichment_system_message.e3.txt").read_text(encoding="utf-8")
    assert e1.vocab_sha256(render_e2()) == e2.EXPECTED_E2_SHA256  # e2 untouched


def test_e3_differs_from_e2_only_in_seed_lines():
    old, new = render_e2().split("\n\n---\n\n"), render_e3().split("\n\n---\n\n")
    assert len(old) == len(new)
    seed_index = [i for i, block in enumerate(old) if block.startswith("Canonical model-type spellings.")]
    assert len(seed_index) == 1
    for i, (a, b) in enumerate(zip(old, new)):
        if i != seed_index[0]:
            assert a == b, f"block {i} changed"
    a_lines, b_lines = old[seed_index[0]].splitlines(), new[seed_index[0]].splitlines()
    assert a_lines[0] == b_lines[0]  # the block's instruction text
    assert all(line.startswith("- ") for line in b_lines[1:])
    assert len(b_lines) > len(a_lines)


def test_e3_vocabulary_is_a_strict_superset_of_e2():
    old = {t["canonical"]: t for t in seed("e2")["terms"]}
    new = {t["canonical"]: t for t in seed("e3")["terms"]}
    assert set(old) < set(new)
    for name, term in old.items():
        assert set(term.get("aliases", [])) <= set(new[name].get("aliases", [])), name
        assert term.get("family_hint") == new[name].get("family_hint"), name
    for name in set(new) - set(old):
        assert new[name].get("family_hint") and new[name].get("ontology_mappings") is not None, name
    branch = "modelling_branch_vocab.json"
    assert (CRITERIA / "e3" / branch).read_bytes() == (CRITERIA / "e2" / branch).read_bytes()


def test_every_new_family_hint_names_existing_family_terms():
    families = {t["label"] for t in json.loads((CRITERIA / "e3" / "modelling_branch_vocab.json").read_text())
                ["fields"]["model_family"]["terms"]}
    old = {t["canonical"] for t in seed("e2")["terms"]}
    for term in seed("e3")["terms"]:
        if term["canonical"] in old:
            continue
        for part in term["family_hint"].split("+"):
            assert part.strip() in families, (term["canonical"], part)


def test_a_drifted_render_is_refused():
    assert e3.e3_problems(render_e3() + " ")


# --- the EBI client ---------------------------------------------------------------------------------

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
    return {"model": SERVED, "choices": [{"message": {"content": "{}", "reasoning_content": "r"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def ebi(session) -> EbiClient:
    return EbiClient(FAKE_KEY, "https://gateway.example/v1/", SERVED, session=session)


def test_ebi_sends_enable_thinking_json_mode_and_the_served_model_only():
    session = FakeSession(FakeResponse(payload=ok_payload()))
    response = ebi(session).chat_completion([{"role": "user", "content": "x"}], tier=zai_client.EBI_TIER,
                                            max_tokens=16000)
    call = session.calls[0]
    assert call["url"] == "https://gateway.example/v1/chat/completions"
    assert call["json"] == {"model": SERVED, "messages": [{"role": "user", "content": "x"}], "max_tokens": 16000,
                            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": True},
                            "response_format": {"type": "json_object"}}
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert response.raw["model"] == SERVED and response.raw["usage"]["prompt_cache_hit_tokens"] == 0


def test_ebi_refuses_other_tiers_missing_config_and_never_echoes_the_key():
    with pytest.raises(ValueError, match="ebi-glm-5.3-flash"):
        ebi(FakeSession(None)).chat_completion([], tier="glm-5.3-flash")
    for kwargs, var in (({"api_key": ""}, "EBI_LLM_API_KEY"), ({"base_url": ""}, "EBI_LLM_BASE_URL"),
                        ({"model": ""}, "EBI_LLM_MODEL")):
        args = {"api_key": FAKE_KEY, "base_url": "https://g/v1", "model": SERVED, **kwargs}
        with pytest.raises(ValueError, match=var):
            EbiClient(**args)
    client = ebi(FakeSession(FakeResponse(status=429, text='{"error":"busy"}')))
    assert FAKE_KEY not in repr(client)
    with pytest.raises(requests.HTTPError) as exc:
        client.chat_completion([], tier=zai_client.EBI_TIER)
    assert FAKE_KEY not in str(exc.value) and "429" in str(exc.value) and "EMBL-EBI" in str(exc.value)


def test_ebi_client_reads_only_the_environment():
    client = EbiClient.from_env({"EBI_LLM_API_KEY": FAKE_KEY, "EBI_LLM_BASE_URL": "https://g/v1",
                                 "EBI_LLM_MODEL": SERVED})
    assert client.base_url == "https://g/v1" and client.model == SERVED


# --- the step ---------------------------------------------------------------------------------------

def answer(**fields):
    base = {"domain_tier1": [], "domain_tier2": [], "domain_tier3": [], "model_use": ["trained in this study"],
            "learning_paradigm": ["supervised"], "model_family": [], "model_type": [], "rationale": "r"}
    return json.dumps({**base, **fields})


class FakeClient:
    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()

    def chat_completion(self, messages, tier, max_tokens=6000, extra_body=None, **kwargs):
        assert tier == zai_client.EBI_TIER
        with self._lock:
            self.messages.append(messages)
        return DeepSeekResponse(content=answer(model_type=["XGBoost"]), reasoning_content=None, prompt_tokens=6000,
                                completion_tokens=500, total_tokens=6500, finish_reason="stop",
                                raw={"model": SERVED, "usage": {"prompt_cache_hit_tokens": 0}})

    def close(self):
        pass


def _input(tmp_path) -> Path:
    path = tmp_path / "input.csv"
    pd.DataFrame([{"record_id": "a", "title": "T", "abstract": "XGBoost.", "journal": "J", "year": "2024"},
                  {"record_id": "b", "title": "T2", "abstract": "SVM.", "journal": "J", "year": "2025"}]
                 ).to_csv(path, index=False)
    return path


def test_step_writes_e3_events_with_the_served_model_and_resumes(tmp_path, monkeypatch):
    client = FakeClient()
    seen = {}
    monkeypatch.setattr(pipeline_steps, "_ebi_client", lambda c: seen.setdefault("concurrency", c) and client)
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)
    monkeypatch.setenv("EBI_LLM_MAX_CONCURRENCY", "7")
    events = tmp_path / "events.csv"
    pipeline_steps.step_llm_classify_enrich_e3(None, input_path=_input(tmp_path), events_out=str(events))
    df = pd.read_csv(events, dtype=str)
    assert list(df.columns) == e3.E3_EVENT_COLUMNS and len(df) == 2
    assert set(df["prompt_version"]) == {"e3"} and set(df["vocab_sha256"]) == {e3.EXPECTED_E3_SHA256}
    assert set(df["model_tier"]) == {"ebi-glm-5.3-flash"} and set(df["model_id"]) == {SERVED}
    assert seen["concurrency"] == 7  # the ramp's measured value, from .env
    assert client.messages[0][0]["content"] == (REPO / "prompts" / "enrichment_system_message.e3.txt").read_text()

    again = FakeClient()
    monkeypatch.setattr(pipeline_steps, "_ebi_client", lambda c: again)
    pipeline_steps.step_llm_classify_enrich_e3(None, input_path=_input(tmp_path), events_out=str(events))
    assert again.messages == [] and len(pd.read_csv(events)) == 2


def test_step_refuses_effort_on_ebi_mixed_logs_and_unknown_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_steps, "_ebi_client", lambda c: FakeClient())
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)
    src = _input(tmp_path)
    with pytest.raises(ValueError, match="reasoning effort"):
        pipeline_steps.step_llm_classify_enrich_e3(None, input_path=src, events_out=str(tmp_path / "x.csv"),
                                                   reasoning_effort="high")
    with pytest.raises(ValueError, match="provider"):
        pipeline_steps.step_llm_classify_enrich_e3(None, provider="deepseek", input_path=src,
                                                   events_out=str(tmp_path / "x.csv"))
    e2_log = tmp_path / "e2_events.csv"
    row = {c: "" for c in e2.E2_EVENT_COLUMNS}
    row.update(record_id="a", prompt_version="e2", parse_status="ok", model_tier="glm-5.3-flash")
    pd.DataFrame([row]).to_csv(e2_log, index=False)
    with pytest.raises(ValueError, match="another prompt version"):
        pipeline_steps.step_llm_classify_enrich_e3(None, input_path=src, events_out=str(e2_log))


def test_enrich_defaults_to_the_latest_prompt_on_ebi():
    import inspect

    from dome_triage import cli

    params = inspect.signature(cli.llm_classify_enrich).parameters
    assert params["prompt"].default.default == "e3" and params["provider"].default.default == "ebi"
    assert "step_llm_classify_enrich_e3" in inspect.getsource(cli.llm_classify_enrich)


def test_new_seed_aliases_normalise_to_their_canonical_terms():
    lookup = e3.render_e3(CRITERIA)[2]
    out = e3.parse_e3(answer(model_type=["ESM-2", "AF2", "nnU-Net", "BiLSTM", "ResNet-50", "GAT", "PINN", "RSF"]), lookup)
    assert out["model_type"] == ["protein language model", "AlphaFold", "U-Net", "long short-term memory", "ResNet",
                                 "graph neural network", "physics-informed neural network", "random survival forest"]
