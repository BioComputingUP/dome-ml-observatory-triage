"""Hand-rolled DeepSeek chat-completions REST client, mirroring `ingest/epmc_client.py`'s
`requests.Session` + `urllib3.util.retry.Retry` pattern -- this codebase's established
minimal-dependency style for an external REST client (no `openai`/`anthropic` SDK in
pyproject.toml, and no reason to add one just for this).

Step 20's central no-RAG guarantee lives here: `chat_completion()` never adds a `tools` key to the
request body unless the caller explicitly passes `enable_search=True` -- the one deliberate,
narrowly-scoped exception used only by the undetermined-subset RAG fallback (see
`llm_classify/runner.py`). See `tests/test_deepseek_client.py` for the test asserting this per
call-type, not as one blanket assertion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_BASE_URL = "https://api.deepseek.com"

# Confirmed 2026-08-20 against the real, official pricing page
# (api-docs.deepseek.com/quick_start/pricing) -- superseding an earlier guess ("deepseek-chat" /
# "deepseek-reasoner") that three unreliable web lookups couldn't verify. Those two legacy names
# turned out to still work as aliases (real calibration calls against them showed genuinely
# different completion-length behavior -- flash averaged ~57 completion tokens/call, the
# "reasoner" alias averaged ~207, consistent with a real reasoning-tier model, not a silent
# fallback to flash), but there's no reason to depend on an alias once the real, current model
# version strings are known: DeepSeek-V4-Flash-0731 / DeepSeek-V4-Pro-0813.
TIER_MODEL_IDS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

# Per-1M-token pricing from the same confirmed pricing page, off-peak cache-miss rates (01:00-04:00
# and 06:00-10:00 UTC are "peak", exactly 2x these rates -- see the pricing page's own footnote).
# NOT used for cost projection (cost_estimator.py deliberately uses only real observed calibration
# spend, never a hardcoded table, since a provider's list price can silently drift and prompt-
# caching on the shared system-message prefix makes real per-call cost noticeably cheaper than
# list price alone would predict -- confirmed live: 16 calibration calls' list-price cost estimate
# came out ~5x the dashboard's actual billed delta). Kept here only as a sanity-check reference for
# a human reading this file, not read by any code path.
TIER_PRICING_USD_PER_1M_TOKENS_OFFPEAK_CACHE_MISS = {
    "flash": {"input": 0.22, "output": 0.66},
    "pro": {"input": 0.66, "output": 1.98},
}


# Low-level connection/protocol failures that occur while consuming an already-"successful"
# response's body -- one layer above what urllib3.Retry (create_session, below) can intercept.
# See chat_completion()'s docstring for the real incident that made this necessary.
_RETRYABLE_NETWORK_EXCEPTIONS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def create_session(
    max_retries: int = 5, backoff_factor: float = 1.0, pool_maxsize: int = 50
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    # pool_maxsize must actually cover runner.py::classify_records' thread-pool width. A session
    # shared across threads *queues* requests past pool_maxsize rather than erroring, so a
    # hardcoded 50 silently capped real concurrency at 50 no matter how high --concurrency went:
    # the extra threads existed but just waited on a connection. Now sized from the caller.
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=max(10, pool_maxsize // 4), pool_maxsize=pool_maxsize
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class DeepSeekResponse:
    content: str
    reasoning_content: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # "stop" (finished normally), "length" (hit max_tokens -- the real, confirmed-live cause of
    # this project's parse_error rows: deepseek-v4-flash's default "thinking mode" can consume the
    # whole token budget before ever writing the JSON answer), or another provider-defined value.
    finish_reason: Optional[str] = None
    raw: dict = field(repr=False, default_factory=dict)


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        timeout: float = 60.0,
        pool_maxsize: int = 50,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = create_session(max_retries, backoff_factor, pool_maxsize)

    def chat_completion(
        self,
        messages: list[dict],
        tier: str,
        max_tokens: int = 6000,
        temperature: float = 0.0,
        request_json_mode: bool = True,
        enable_search: bool = False,
        max_network_retries: int = 3,
        extra_body: Optional[dict] = None,
    ) -> DeepSeekResponse:
        """POSTs one chat-completions call. `enable_search=True` is the ONLY way this method ever
        adds a `tools` key to the request body -- used exclusively by the Fallback-B RAG-assisted
        re-ask on the small undetermined subset (see runner.py), never the primary or forced-guess
        calls. The exact tool schema DeepSeek expects for web search is unverified (the same
        unreliable research that couldn't pin down pricing/model names also couldn't confirm this)
        -- check platform.deepseek.com/docs before relying on this path; if the shape below is
        wrong, DeepSeek will reject the request with a clear 4xx, not silently ignore it, because
        of the raise-on-non-2xx behavior below.

        Raises `requests.HTTPError` with the real response body attached on any non-2xx status --
        a wrong TIER_MODEL_IDS entry or a malformed tool schema must fail loudly, not silently.

        Retries up to `max_network_retries` times (1s/2s/4s backoff) on a transient, low-level
        network/protocol failure -- connection reset, or the response getting cut off mid-stream.
        A real, confirmed-live incident this fixes: a 39-minute, 840-call real paid run died on
        exactly `requests.exceptions.ChunkedEncodingError` ("Response ended prematurely") with no
        retry at all. `create_session`'s `urllib3.Retry` above does NOT catch this class of error
        -- it wraps the connect/HTTP-status layer, but this exception surfaces one layer up, while
        `requests` is consuming an already-"successful" response's streamed body (confirmed from
        the real traceback: raised inside `Session.send`'s `if not stream: r.content`, called by
        `session.post()` itself). Only an explicit retry around the whole request call, as done
        here, catches it. HTTP-status errors (4xx/5xx, raised below) are NOT retried here --
        urllib3's Retry already handles the retryable 5xx set, and a 4xx is never transient.

        `max_tokens=6000` (raised from an earlier, too-low 400, then a still-conservative 2000): a
        real, confirmed-live incident -- 84/1000 real paid calls on `deepseek-v4-flash` came back
        as unparseable `parse_error` rows, traced directly to their raw content being cut off
        mid-JSON (e.g. `'{"classification":'`) or entirely empty. `deepseek-v4-flash` defaults to
        "thinking mode" (per the confirmed pricing page) and can spend its whole token budget on
        internal reasoning before ever writing the JSON answer. Raising the cap doesn't cost more
        unless the model actually uses the extra tokens -- billing is per token generated, not per
        the ceiling -- so a generous cap only removes the premature-cutoff risk, it doesn't inflate
        cost on the (large majority of) calls that finish well under it. `finish_reason` (below) is
        captured on every response specifically so any future truncation is diagnosable directly
        from real data, not inferred from the failure pattern the way this one had to be."""
        if tier not in TIER_MODEL_IDS:
            raise ValueError(f"Unknown tier {tier!r} -- expected one of {list(TIER_MODEL_IDS)}")

        body: dict = {
            "model": TIER_MODEL_IDS[tier],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if request_json_mode:
            body["response_format"] = {"type": "json_object"}
        if enable_search:
            body["tools"] = [{"type": "web_search"}]
        if extra_body:
            # Additive escape hatch for provider params this client doesn't model, e.g.
            # {"thinking": {"type": "disabled"}} -- CONFIRMED LIVE (2026-08-27, real API probe):
            # that exact shape drops reasoning_tokens from ~322 to 0 on deepseek-v4-flash with an
            # identical final answer, the fix for thinking-mode output tokens being ~90% of the
            # Step 20j enrichment run's real cost. ("reasoning_effort": "none" also works;
            # "enable_thinking" is silently ignored -- probed, not assumed.)
            body.update(extra_body)

        resp = None
        for attempt in range(max_network_retries + 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout,
                )
                break
            except _RETRYABLE_NETWORK_EXCEPTIONS:
                if attempt >= max_network_retries:
                    raise
                time.sleep(2**attempt)

        if not resp.ok:
            raise requests.HTTPError(
                f"DeepSeek API error {resp.status_code} for model={body['model']}: {resp.text[:2000]}",
                response=resp,
            )
        data = resp.json()
        choice_wrapper = data["choices"][0]
        choice = choice_wrapper["message"]
        usage = data.get("usage", {})
        return DeepSeekResponse(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            finish_reason=choice_wrapper.get("finish_reason"),
            raw=data,
        )

    def close(self) -> None:
        self.session.close()
