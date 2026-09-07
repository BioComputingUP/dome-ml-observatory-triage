"""NCBI E-utilities client -- the *second* metadata source, deliberately not Europe PMC.

Why a different provider at all: the bulk pool was built entirely from Europe PMC
(`ingest/bulk_match.py`, `resultType=core`). Re-asking EPMC for the records whose abstract EPMC
never returned in the first place mostly returns the same gap again -- confirmed live 2026-08-27,
where an EPMC `get_by_ids` re-fetch pass over the same ids was slow and largely redundant. NCBI
PubMed/PMC is the upstream source EPMC itself mirrors, with independent record coverage, so it is
a genuinely different question rather than the same one asked twice.

Rate limits are NCBI's published ones: 3 requests/sec anonymous, 10/sec with an API key
(`NCBI_API_KEY`). Everything here is batched (200 ids per call, POST so the id list never hits a
URL length limit) and rate-limited by a shared token bucket, so raising concurrency past the limit
is pointless by construction -- the bucket, not the worker count, sets the real pace.
"""

from __future__ import annotations

import os
import threading
import time
import xml.etree.ElementTree as ET
from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# NCBI's own documented ceiling for an efetch id list; also the id-converter's documented max.
BATCH_SIZE = 200
# The converter is GET-only, and DOIs are long and percent-encode heavily -- 200 of them builds a
# multi-kilobyte URL. Kept small so a batch can never fail purely on URL length.
DOI_BATCH_SIZE = 50

_RATE_WITH_KEY = 10.0
_RATE_ANONYMOUS = 3.0


class _RateLimiter:
    """Shared token bucket. Threads block here rather than sleeping a fixed interval, so a slow
    response naturally lets the next thread go immediately instead of stacking dead time."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self.min_interval


def _text(el: Optional[ET.Element]) -> Optional[str]:
    """Full text of an element including nested markup (PubMed wraps <i>/<sup> inside titles and
    abstract paragraphs -- `.text` alone silently truncates at the first child tag)."""
    if el is None:
        return None
    value = "".join(el.itertext()).strip()
    return value or None


class NcbiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
        timeout: int = 60,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY") or None
        self.timeout = timeout
        self.limiter = _RateLimiter(_RATE_WITH_KEY if self.api_key else _RATE_ANONYMOUS)
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)

    @property
    def rate_per_sec(self) -> float:
        return _RATE_WITH_KEY if self.api_key else _RATE_ANONYMOUS

    def _common_params(self) -> dict:
        params = {"tool": "dome-triage"}
        # NCBI asks for a contact address but does not require one; only ever sent when the
        # operator explicitly opts in via NCBI_EMAIL, never inferred from git/user config.
        email = os.environ.get("NCBI_EMAIL")
        if email:
            params["email"] = email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def convert_ids(self, ids: list[str]) -> dict[str, dict]:
        """PMCID -> PMID (or DOI -> PMID) via the PMC ID Converter. Returns {input_id: {pmid,
        pmcid, doi}} for whatever resolved; unresolved ids are simply absent (never raises on a
        partial miss).

        **Every id in one call must be the SAME type** -- confirmed live 2026-08-27: a batch mixing
        PMCIDs and DOIs is rejected outright with `400 Bad Request`, taking the whole batch down
        rather than just the odd id. The converter sniffs the id type from the batch, so callers
        must bucket by type first (see `metadata_repair.fetch_repairs`)."""
        if not ids:
            return {}
        params = self._common_params()
        params.update({"ids": ",".join(ids), "format": "json", "versions": "no"})
        self.limiter.acquire()
        resp = self.session.get(IDCONV_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        out: dict[str, dict] = {}
        # The API returns pmid as a JSON *number*; the caller's ids are strings, so every
        # comparison and key must be normalized or a pmid-keyed lookup silently matches nothing.
        wanted = {str(value) for value in ids}
        for record in resp.json().get("records", []):
            if record.get("status") == "error":
                continue
            mapped = {
                "pmid": str(record["pmid"]) if record.get("pmid") else None,
                "pmcid": record.get("pmcid") or None,
                "doi": record.get("doi") or None,
            }
            if not any(mapped.values()):
                continue
            # Key by whichever input form this record answers, so the caller can match it back.
            for key in ("pmcid", "doi", "pmid"):
                value = mapped.get(key)
                if value and value in wanted:
                    out[value] = mapped
        return out

    def efetch_pubmed(self, pmids: list[str]) -> dict[str, dict]:
        """Batch efetch from PubMed. Returns {pmid: {title, abstract, journal, year, authors, doi}}
        for whichever pmids PubMed knows; missing ones are absent."""
        if not pmids:
            return {}
        data = self._common_params()
        data.update({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        self.limiter.acquire()
        resp = self.session.post(f"{EUTILS_BASE}/efetch.fcgi", data=data, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_pubmed_xml(resp.content)

    @staticmethod
    def _parse_pubmed_xml(payload: bytes) -> dict[str, dict]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return {}
        out: dict[str, dict] = {}
        for article in root.iter("PubmedArticle"):
            citation = article.find("MedlineCitation")
            if citation is None:
                continue
            pmid = _text(citation.find("PMID"))
            if not pmid:
                continue
            art = citation.find("Article")
            if art is None:
                continue

            # A structured abstract is several labelled <AbstractText> parts; join them in order
            # with their labels so nothing is silently dropped to just the first section.
            parts: list[str] = []
            for node in art.iter("AbstractText"):
                body = _text(node)
                if not body:
                    continue
                label = node.attrib.get("Label")
                parts.append(f"{label}: {body}" if label else body)

            journal = art.find("Journal")
            year = None
            if journal is not None:
                pub_date = journal.find("./JournalIssue/PubDate")
                if pub_date is not None:
                    year = _text(pub_date.find("Year")) or _text(pub_date.find("MedlineDate"))
                    if year and len(year) >= 4 and year[:4].isdigit():
                        year = year[:4]
                    else:
                        year = None

            authors = None
            author_list = art.find("AuthorList")
            if author_list is not None:
                names = []
                for author in author_list.findall("Author"):
                    last = _text(author.find("LastName"))
                    initials = _text(author.find("Initials"))
                    if last:
                        names.append(f"{last} {initials}" if initials else last)
                    elif (collective := _text(author.find("CollectiveName"))):
                        names.append(collective)
                if names:
                    authors = ", ".join(names) + "."

            doi = None
            pmcid = None
            for article_id in article.iter("ArticleId"):
                id_type = article_id.attrib.get("IdType")
                if id_type == "doi" and doi is None:
                    doi = _text(article_id)
                elif id_type == "pmc" and pmcid is None:
                    pmcid = _text(article_id)

            out[pmid] = {
                "pmid": pmid,
                "title": _text(art.find("ArticleTitle")),
                "abstract": "\n".join(parts) if parts else None,
                "journal": _text(journal.find("Title")) if journal is not None else None,
                "year": year,
                "authors": authors,
                "doi": doi,
                "pmcid": pmcid,
            }
        return out

    def close(self) -> None:
        self.session.close()


def chunked(values: Iterable[str], size: int = BATCH_SIZE) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
