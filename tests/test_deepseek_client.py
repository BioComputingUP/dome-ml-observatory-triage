import pytest
import requests

from dome_triage.llm_classify.deepseek_client import DeepSeekClient, TIER_MODEL_IDS


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(json_data)

    def json(self) -> dict:
        return self._json_data


def _fake_success_response() -> dict:
    return {
        "choices": [{"message": {"content": '{"classification": "positive", "rationale": "x"}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def test_chat_completion_default_max_tokens_is_6000_not_400(monkeypatch):
    """Regression test for a real, confirmed-live incident: max_tokens=400 was too low for
    deepseek-v4-flash's default "thinking mode" and truncated 84/1000 real paid responses
    mid-JSON. Locks in the raised default so it can't silently regress."""
    client = DeepSeekClient(api_key="sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert captured["max_tokens"] == 6000


def test_chat_completion_captures_finish_reason(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")

    def fake_post(url, headers=None, json=None, timeout=None):
        response = _fake_success_response()
        response["choices"][0]["finish_reason"] = "length"
        return _FakeResponse(response)

    monkeypatch.setattr(client.session, "post", fake_post)

    response = client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert response.finish_reason == "length"


def test_chat_completion_finish_reason_defaults_to_none_when_absent(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    monkeypatch.setattr(
        client.session, "post", lambda url, headers=None, json=None, timeout=None: _FakeResponse(_fake_success_response())
    )

    response = client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert response.finish_reason is None


def test_chat_completion_never_includes_tools_key_by_default(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_chat_completion_forced_guess_style_call_also_omits_tools(monkeypatch):
    """Even with enable_search left at its default False for a non-primary call shape (the
    forced-guess fallback), the tools key must never appear -- only an explicit
    enable_search=True call (Fallback B) may ever set it."""
    client = DeepSeekClient(api_key="sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="flash", enable_search=False)

    assert "tools" not in captured


def test_chat_completion_enable_search_true_is_the_only_way_to_add_tools(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="flash", enable_search=True)

    assert "tools" in captured


def test_chat_completion_uses_the_correct_model_id_per_tier(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="pro")

    assert captured["model"] == TIER_MODEL_IDS["pro"]


def test_chat_completion_unknown_tier_raises_value_error():
    client = DeepSeekClient(api_key="sk-test")
    with pytest.raises(ValueError):
        client.chat_completion([{"role": "user", "content": "hi"}], tier="ultra")


def test_chat_completion_parses_real_usage_and_content(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    monkeypatch.setattr(
        client.session, "post", lambda url, headers=None, json=None, timeout=None: _FakeResponse(_fake_success_response())
    )

    response = client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.total_tokens == 120
    assert "positive" in response.content


def test_chat_completion_raises_http_error_with_body_on_non_2xx(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    monkeypatch.setattr(
        client.session,
        "post",
        lambda url, headers=None, json=None, timeout=None: _FakeResponse({"error": "bad model"}, status_code=404),
    )

    with pytest.raises(requests.HTTPError, match="bad model"):
        client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")


def test_chat_completion_retries_on_chunked_encoding_error_and_succeeds(monkeypatch):
    """Regression test for a real, confirmed-live incident: a 39-minute, 840-call real paid run
    died on exactly this exception with no retry, losing everything already completed. This locks
    in that a transient ChunkedEncodingError is now retried, not fatal."""
    client = DeepSeekClient(api_key="sk-test")
    monkeypatch.setattr(client, "timeout", 0)  # irrelevant to the fake, just avoids any real wait
    calls = {"n": 0}

    def flaky_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", flaky_post)
    monkeypatch.setattr("dome_triage.llm_classify.deepseek_client.time.sleep", lambda seconds: None)

    response = client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert calls["n"] == 3
    assert "positive" in response.content


def test_chat_completion_raises_after_exhausting_network_retries(monkeypatch):
    client = DeepSeekClient(api_key="sk-test")
    monkeypatch.setattr(client.session, "post", lambda *a, **kw: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("connection reset")
    ))
    monkeypatch.setattr("dome_triage.llm_classify.deepseek_client.time.sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        client.chat_completion([{"role": "user", "content": "hi"}], tier="flash", max_network_retries=2)


def test_chat_completion_does_not_retry_http_error_status(monkeypatch):
    """A 4xx/5xx HTTP status is a completed response, not a network failure -- must not be
    silently retried by the network-error retry loop (urllib3's own Retry already handles the
    retryable 5xx set at the connection layer; a non-retryable status should surface immediately)."""
    client = DeepSeekClient(api_key="sk-test")
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse({"error": "bad request"}, status_code=400)

    monkeypatch.setattr(client.session, "post", fake_post)

    with pytest.raises(requests.HTTPError):
        client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert calls["n"] == 1


def test_chat_completion_sends_authorization_bearer_header(monkeypatch):
    client = DeepSeekClient(api_key="sk-super-secret")
    captured_headers = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured_headers.update(headers)
        return _FakeResponse(_fake_success_response())

    monkeypatch.setattr(client.session, "post", fake_post)

    client.chat_completion([{"role": "user", "content": "hi"}], tier="flash")

    assert captured_headers["Authorization"] == "Bearer sk-super-secret"


def test_connection_pool_is_sized_to_the_requested_concurrency():
    # Real, confirmed limitation: pool_maxsize was hardcoded at 50, so a session shared across a
    # wider thread pool silently QUEUED requests past 50 rather than erroring -- --concurrency 200
    # produced 200 threads but only 50 real concurrent connections. The pool must track the width.
    from dome_triage.llm_classify.deepseek_client import DeepSeekClient

    client = DeepSeekClient(api_key="test-key", pool_maxsize=200)
    adapter = client.session.get_adapter("https://api.deepseek.com")
    assert adapter._pool_maxsize == 200


def test_connection_pool_keeps_the_historical_default_when_unspecified():
    from dome_triage.llm_classify.deepseek_client import DeepSeekClient

    client = DeepSeekClient(api_key="test-key")
    adapter = client.session.get_adapter("https://api.deepseek.com")
    assert adapter._pool_maxsize == 50
