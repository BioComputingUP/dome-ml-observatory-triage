"""Z.ai chat-completions client for GLM-5.3 / GLM-5.3-Flash, with the same `chat_completion()`
signature as `DeepSeekClient`, so the enrichment loops run on it unchanged.

Prompt e2 was validated on this client (glm5.3_test repository, 2026-11-05/16; see
`curation_criteria/e2/PROVENANCE.md`). It is used for enrichment only; classification stays on the
validated DeepSeek path.

What differs from DeepSeek, and why (https://docs.z.ai/api-reference/llm/chat-completion):

- Base URL `https://api.z.ai/api/paas/v4`: the pay-as-you-go API. The GLM Coding Plan's endpoint
  is a different product and is never used here.
- GLM-5.3 and GLM-5.3-Flash think unconditionally (`thinking.type` can only be "enabled"). Depth is
  a TOP-LEVEL `reasoning_effort`, which Z.ai defaults to "max". The validated runs used "high", so
  that is the default here. "max" was measured on 100 papers: the same tag quality, about 6x the
  reasoning tokens, and one record truncated at 16,000 tokens.
- Usage reports cached prompt tokens as `prompt_tokens_details.cached_tokens`. The event loops read
  DeepSeek's `prompt_cache_hit_tokens`, so the response is normalised to carry that name too;
  otherwise every event would silently record a cache share of 0.
- Thinking at 16,000 output tokens can outlast DeepSeek's 60 s timeout, and a timed-out request
  may still be billed, so the default timeout is 300 s.

No tools are ever sent: `enable_search=True` raises rather than adding a key the validated
configuration never had. The API key is never printed (`__repr__`).
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from dome_triage.llm_classify.deepseek_client import (
    _RETRYABLE_NETWORK_EXCEPTIONS,
    DeepSeekResponse,
    create_session,
)

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
PROVIDER = "zai"

TIER_MODEL_IDS = {
    "glm-5.3-flash": "glm-5.3-flash",
    "glm-5.3": "glm-5.3",
}

DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")

# List prices, USD per 1M tokens, read 2026-10-25 from https://docs.z.ai/guides/overview/pricing.
# For a person reading this file and for the pre-flight projection only. The billed figure on the
# Z.ai dashboard is the authority.
TIER_PRICING_USD_PER_1M_TOKENS = {
    "glm-5.3-flash": {"input_cache_hit": 0.03, "input_cache_miss": 0.15, "output": 0.50},
    "glm-5.3": {"input_cache_hit": 0.26, "input_cache_miss": 1.40, "output": 4.40},
}
# Measured e2 token profile on GLM-5.3-Flash, effort high: wave2, 250 records, 2026-11-16
# (cache hit / cache miss / output tokens per record). The prompt-cache share varies by run (15-77%).
E2_TOKEN_PROFILE = {"input_cache_hit": 829, "input_cache_miss": 4759, "output": 441}


def projected_usd(tier: str, n_records: int, headroom: float = 1.5) -> float:
    """List-price projection from the measured e2 profile, with headroom for a longer-thinking batch."""
    p, t = TIER_PRICING_USD_PER_1M_TOKENS[tier], E2_TOKEN_PROFILE
    per = (t["input_cache_hit"] * p["input_cache_hit"] + t["input_cache_miss"] * p["input_cache_miss"]
           + t["output"] * p["output"]) / 1e6
    return n_records * per * headroom


def is_zai_tier(tier: str) -> bool:
    return tier in TIER_MODEL_IDS


class ZaiClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_retries: int = 8,
        backoff_factor: float = 1.0,
        timeout: float = 300.0,
        pool_maxsize: int = 50,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("ZAI_API_KEY is empty -- add it to the repo-root .env (gitignored); see .env.example")
        if reasoning_effort not in REASONING_EFFORT_LEVELS:
            raise ValueError(f"reasoning_effort must be one of {list(REASONING_EFFORT_LEVELS)}, got {reasoning_effort!r}")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.session = session if session is not None else create_session(max_retries, backoff_factor, pool_maxsize)

    def __repr__(self) -> str:  # never show the key
        return f"ZaiClient(base_url={self.base_url!r}, reasoning_effort={self.reasoning_effort!r})"

    def build_body(self, messages: list[dict], tier: str, max_tokens: int, temperature: float,
                   request_json_mode: bool) -> dict:
        if tier not in TIER_MODEL_IDS:
            raise ValueError(f"Unknown tier {tier!r} -- expected one of {list(TIER_MODEL_IDS)}")
        body: dict = {
            "model": TIER_MODEL_IDS[tier],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {"type": "enabled"},
            "reasoning_effort": self.reasoning_effort,
        }
        if request_json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

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
        if enable_search:
            raise ValueError("enable_search is never used for enrichment; no tools are sent")
        if extra_body:
            # DeepSeek's thinking shape has no meaning here; Z.ai's effort is set on the client.
            raise ValueError("pass reasoning_effort to ZaiClient, not extra_body")
        body = self.build_body(messages, tier, max_tokens, temperature, request_json_mode)

        resp = None
        for attempt in range(max_network_retries + 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
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
                f"Z.ai API error {resp.status_code} for model={body['model']}: {resp.text[:2000]}",
                response=resp,
            )
        data = normalise_usage(resp.json())
        choice_wrapper = data["choices"][0]
        choice = choice_wrapper["message"]
        usage = data["usage"]
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


def normalise_usage(data: dict) -> dict:
    """Adds DeepSeek's `prompt_cache_hit_tokens` from Z.ai's `prompt_tokens_details.cached_tokens`.
    `completion_tokens_details.reasoning_tokens` passes through when Z.ai sends it; when it does
    not, events record 0."""
    usage = dict(data.get("usage") or {})
    if "prompt_cache_hit_tokens" not in usage:
        usage["prompt_cache_hit_tokens"] = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    data["usage"] = usage
    return data
