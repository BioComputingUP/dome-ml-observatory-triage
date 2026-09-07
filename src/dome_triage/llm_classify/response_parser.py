"""Parses a DeepSeek chat-completion response into a classification -- never raises, never
silently coerces an unparseable response into a guessed label. `PARSE_ERROR` rows are excluded
from Cohen's-kappa scoring entirely (see `reporting/agreement.py`) and reported as their own
visible count, not folded into any real class. See `tests/test_response_parser.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

VALID_CLASSIFICATIONS = ("positive", "negative", "undeterminable")
PARSE_ERROR = "parse_error"

# Reasoning-tier models often emit a <think>...</think> block even outside a dedicated
# reasoning_content field -- stripped before a second JSON-parse attempt.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BRACE_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CLASSIFICATION_WORD_RE = re.compile(r"\b(positive|negative|undeterminable)\b", re.IGNORECASE)


@dataclass
class ParsedResult:
    classification: str
    rationale: str
    parse_fallback_used: bool


def _try_json_dict(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_from_dict(data: dict) -> Optional[tuple[str, str]]:
    classification = str(data.get("classification", "")).strip().lower()
    if classification in VALID_CLASSIFICATIONS:
        return classification, str(data.get("rationale", "")).strip()
    return None


def parse_classification(raw_content: str) -> ParsedResult:
    text = (raw_content or "").strip()

    # 1. Clean JSON.
    data = _try_json_dict(text)
    if data is not None:
        extracted = _extract_from_dict(data)
        if extracted is not None:
            return ParsedResult(extracted[0], extracted[1], parse_fallback_used=False)

    # 2. Strip a <think>...</think> prefix and retry.
    stripped = _THINK_BLOCK_RE.sub("", text).strip()
    if stripped != text:
        data = _try_json_dict(stripped)
        if data is not None:
            extracted = _extract_from_dict(data)
            if extracted is not None:
                return ParsedResult(extracted[0], extracted[1], parse_fallback_used=True)
        text = stripped

    # 3. JSON wrapped in surrounding prose (e.g. "Here is my answer: {...}").
    brace_match = _BRACE_BLOCK_RE.search(text)
    if brace_match:
        data = _try_json_dict(brace_match.group(0))
        if data is not None:
            extracted = _extract_from_dict(data)
            if extracted is not None:
                return ParsedResult(extracted[0], extracted[1], parse_fallback_used=True)

    # 4. Final fallback: first whole-word classification term anywhere in the text.
    word_match = _CLASSIFICATION_WORD_RE.search(text)
    if word_match:
        return ParsedResult(word_match.group(1).lower(), text[:300], parse_fallback_used=True)

    return ParsedResult(PARSE_ERROR, text[:300], parse_fallback_used=True)
