"""Crossref REST client -- the third and last abstract source, tried only on DOIs that NCBI could
not resolve an abstract for.

Why it is worth a pass at all: Crossref holds whatever abstract the *publisher* deposited, which is
a genuinely different set from what PubMed indexes. Coverage is patchy (many publishers deposit no
abstract at all), so this is a recovery pass over the remainder, not a primary source.

Throughput: Crossref has no batch endpoint, so this is concurrent single-DOI GETs. Setting
`CROSSREF_MAILTO` puts requests in Crossref's "polite pool", which is both faster and the
behaviour they ask for. The limiter reads Crossref's own `X-Rate-Limit-Limit` /
`X-Rate-Limit-Interval` response headers and adapts to whatever they actually advertise rather than
guessing a fixed number.
"""

from __future__ import annotations

import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CROSSREF_BASE = "https://api.crossref.org/works"

# Real, confirmed 2026-08-27: opening at 50/s anonymously got 429ed almost immediately and the run
# recovered nothing. The anonymous pool is genuinely strict; the polite pool (set CROSSREF_MAILTO)
# is what Crossref asks for and what actually sustains throughput. Start conservative for each and
# let `X-Rate-Limit-*` headers raise it and 429s lower it.
# Crossref's own headers report 3 req/s; batching 50 DOIs per request is what turns that into
# ~150 DOIs/sec rather than 3.
_POLITE_START_RATE = 3.0
_ANONYMOUS_START_RATE = 2.0
CROSSREF_BATCH_SIZE = 50
_MIN_RATE = 1.0
_JATS_TAG = re.compile(r"<[^>]+>")


class _AdaptiveRateLimiter:
    """Token bucket whose rate can be revised at runtime from Crossref's own headers."""

    def __init__(self, rate_per_sec: float, ceiling_per_sec: Optional[float] = None):
        self._lock = threading.Lock()
        self._interval = 1.0 / rate_per_sec
        # Fastest the limiter may ever go, raised if Crossref's headers advertise more.
        self._floor_interval = 1.0 / (ceiling_per_sec or rate_per_sec)
        self._next_at = 0.0

    def set_rate(self, rate_per_sec: float) -> None:
        """Raise the ceiling to whatever Crossref actually advertises, and open up to it."""
        if rate_per_sec <= 0:
            return
        with self._lock:
            self._floor_interval = min(self._floor_interval, 1.0 / rate_per_sec)

    def back_off(self, factor: float = 2.0) -> None:
        """Halve the shared rate on a 429 so every worker slows together, floored so the run
        cannot grind to a standstill."""
        with self._lock:
            self._interval = min(self._interval * factor, 1.0 / _MIN_RATE)

    def recover(self, factor: float = 1.05) -> None:
        """Additive-increase half of AIMD. Without this the limiter only ever ratchets *down*: a
        brief burst of 429s early on permanently caps the rest of a multi-hour run (real, measured
        2026-08-27 -- a 400-DOI batch settled at 3.65/s and never climbed back, which would have
        made the full 49k-DOI pass ~3.7 hours). Creeping back up finds the sustainable rate instead
        of assuming the worst moment is the truth."""
        with self._lock:
            self._interval = max(self._interval / factor, self._floor_interval)

    @property
    def rate_per_sec(self) -> float:
        return 1.0 / self._interval

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self._interval


def strip_jats(raw: Optional[str]) -> Optional[str]:
    """Crossref abstracts are JATS XML fragments (`<jats:p>`, `<jats:sec>`, `<jats:title>`).
    Parse properly where possible so nested markup is flattened rather than mangled, and fall back
    to a tag strip for the fragments that are not well-formed standalone XML."""
    if not raw:
        return None
    text = raw.strip()
    try:
        wrapped = f"<root xmlns:jats='http://jats.nlm.nih.gov'>{text}</root>"
        parsed = " ".join(part.strip() for part in ET.fromstring(wrapped).itertext())
    except ET.ParseError:
        parsed = _JATS_TAG.sub(" ", text)
    cleaned = " ".join(parsed.split())
    # Publishers very commonly deposit a bare "Abstract" heading and nothing else -- that is a
    # non-answer, and letting it through would count as a repair while telling the classifier
    # nothing.
    if not cleaned or cleaned.lower() in {"abstract", "summary"}:
        return None
    return cleaned


class CrossrefClient:
    def __init__(
        self,
        mailto: Optional[str] = None,
        rate_per_sec: Optional[float] = None,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        timeout: int = 30,
    ):
        # Only ever sent when the operator explicitly opts in; never inferred from git/user config.
        self.mailto = mailto if mailto is not None else os.environ.get("CROSSREF_MAILTO") or None
        self.timeout = timeout
        if rate_per_sec is None:
            rate_per_sec = _POLITE_START_RATE if self.mailto else _ANONYMOUS_START_RATE
        self.limiter = _AdaptiveRateLimiter(rate_per_sec, ceiling_per_sec=rate_per_sec)
        self.throttled = 0
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
        self.session.mount("https://", adapter)
        self._headers = {
            "User-Agent": (
                f"dome-triage/1.0 (mailto:{self.mailto})" if self.mailto else "dome-triage/1.0"
            )
        }

    def _observe_rate_headers(self, resp: requests.Response) -> None:
        limit = resp.headers.get("X-Rate-Limit-Limit")
        interval = resp.headers.get("X-Rate-Limit-Interval")
        if not limit or not interval:
            return
        try:
            seconds = float(str(interval).rstrip("s") or 1)
            if seconds > 0:
                self.limiter.set_rate(float(limit) / seconds)
        except ValueError:
            return

    def get_abstracts_batch(self, dois: list[str]) -> dict[str, dict]:
        """Batch lookup via Crossref's `filter=doi:a,doi:b,...` (repeated filters of the same field
        are OR'd). Returns {doi: payload} for those with a usable abstract; DOIs Crossref does not
        know, or knows without an abstract, are simply absent.

        This is what makes the pass viable at all: Crossref enforces ~3 requests/sec, so one
        request per DOI meant 4.5 DOIs/sec and ~3 hours for the real 49k-DOI remainder (measured,
        2026-08-27). At 50 DOIs per request the same ceiling yields ~150 DOIs/sec.

        Returned DOIs are matched back case-insensitively -- Crossref normalizes DOIs to lowercase,
        so a pool DOI containing uppercase would otherwise silently never match its own result."""
        if not dois:
            return {}
        self.limiter.acquire()
        params = {
            "filter": ",".join(f"doi:{doi}" for doi in dois),
            "select": "DOI,abstract,title,container-title,issued",
            "rows": len(dois),
        }
        resp = self.session.get(
            CROSSREF_BASE, params=params, headers=self._headers, timeout=self.timeout
        )
        self._observe_rate_headers(resp)
        if resp.status_code != 429:
            self.limiter.recover()
        if resp.status_code == 429:
            self.throttled += 1
            self.limiter.back_off()
            raise requests.HTTPError(f"429 rate-limited on batch of {len(dois)}", response=resp)
        resp.raise_for_status()

        wanted = {doi.lower(): doi for doi in dois}
        out: dict[str, dict] = {}
        for item in (resp.json().get("message") or {}).get("items", []):
            returned = (item.get("DOI") or "").lower()
            original = wanted.get(returned)
            if not original:
                continue
            abstract = strip_jats(item.get("abstract"))
            if not abstract:
                continue
            titles = item.get("title") or []
            containers = item.get("container-title") or []
            year = None
            issued = (item.get("issued") or {}).get("date-parts") or []
            if issued and issued[0] and issued[0][0]:
                year = str(issued[0][0])
            out[original] = {
                "doi": original,
                "abstract": abstract,
                "title": titles[0] if titles else None,
                "journal": containers[0] if containers else None,
                "year": year,
            }
        return out

    def get_abstract(self, doi: str) -> Optional[dict]:
        """Returns {'doi', 'abstract', 'title', 'journal', 'year'} when Crossref has a usable
        abstract for this DOI, else None. A 404 is a normal 'not deposited', not an error."""
        self.limiter.acquire()
        url = f"{CROSSREF_BASE}/{quote(doi, safe='')}"
        resp = self.session.get(url, headers=self._headers, timeout=self.timeout)
        self._observe_rate_headers(resp)
        if resp.status_code != 429:
            self.limiter.recover()
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            # Still throttled after urllib3 exhausted its own Retry-After-aware retries: back the
            # whole client off so every worker slows, not just this one, then let the caller mark
            # this DOI as a transient error so a resume picks it up again.
            self.throttled += 1
            self.limiter.back_off()
            raise requests.HTTPError(f"429 rate-limited on {doi}", response=resp)
        resp.raise_for_status()
        message = resp.json().get("message") or {}
        abstract = strip_jats(message.get("abstract"))
        if not abstract:
            return None
        titles = message.get("title") or []
        containers = message.get("container-title") or []
        year = None
        issued = (message.get("issued") or {}).get("date-parts") or []
        if issued and issued[0] and issued[0][0]:
            year = str(issued[0][0])
        return {
            "doi": doi,
            "abstract": abstract,
            "title": titles[0] if titles else None,
            "journal": containers[0] if containers else None,
            "year": year,
        }

    def close(self) -> None:
        self.session.close()
