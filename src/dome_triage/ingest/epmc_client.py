"""Europe PMC REST API client.

Ported from DOME-Copilot-Data-Analysis/EPMC_growth_graph/fetch_epmc_growth_data.py, which already
proved out the cursorMark deep-pagination + urllib3 Retry pattern against this exact API. Extended
here with a batch metadata lookup (`get_by_ids`) used by `ingest enrich-metadata` to fill in
title/abstract for the id_pair_only and pdf_directory_gold source rows, which only carry IDs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

DEFAULT_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_BATCH_CHUNK_SIZE = 40  # keep query URLs well under length limits


def create_session(max_retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class EpmcClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        page_size: int = 100,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.session = create_session(max_retries, backoff_factor)

    def search(
        self,
        query: str,
        result_type: str = "core",
        page_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Yield every result for `query` via cursorMark deep pagination. `show_progress` shows a
        live tqdm bar against the API's own `hitCount` (known after the first page) -- turn this
        on for large human-triggered bulk fetches, off for small internal lookups."""
        cursor = "*"
        page_size = page_size or self.page_size
        pbar = tqdm(total=None, unit="records", desc=query[:40], disable=not show_progress)
        try:
            while True:
                params = {
                    "query": query,
                    "pageSize": page_size,
                    "cursorMark": cursor,
                    "format": "json",
                    "resultType": result_type,
                }
                resp = self.session.get(f"{self.base_url}/search", params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                if pbar.total is None:
                    pbar.total = data.get("hitCount", 0)

                results = data.get("resultList", {}).get("result", [])
                if not results:
                    return
                yield from results
                pbar.update(len(results))

                next_cursor = data.get("nextCursorMark", "")
                if not next_cursor or next_cursor == cursor:
                    return
                cursor = next_cursor
        finally:
            pbar.close()

    def count(self, query: str) -> int:
        """Cheap count-only lookup: one HTTP request, pageSize=1, resultType=idlist (no full
        metadata for even the single sample row) -- purely to read the API's own `hitCount`, not
        to iterate results. Used for the AI-only/ML-only/combined breakdown counts in
        ingest/bulk_match.py, so getting that breakdown never requires a second full fetch."""
        params = {"query": query, "pageSize": 1, "format": "json", "resultType": "idlist"}
        resp = self.session.get(f"{self.base_url}/search", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json().get("hitCount", 0)

    def get_by_ids(
        self,
        ids: list[str],
        id_type: Literal["pmcid", "pmid", "doi"],
        show_progress: bool = False,
    ) -> dict[str, dict]:
        """Batch metadata lookup. Returns {id: result_dict} for whichever ids were found;
        missing ids are simply absent from the returned dict (never raises for a partial miss).

        **Real, confirmed bug fixed here (found live while building Step 19c, verified directly
        against the real EPMC API, not a hypothetical):** for `id_type="pmid"`, a *quoted* value
        combined with the trailing `AND SRC:MED` silently returns zero hits whenever that chunk
        has exactly one id -- `(EXT_ID:"31501885") AND SRC:MED` returns 0 live hits even though
        `EXT_ID:"31501885"` alone (no SRC:MED) correctly finds the record, and
        `(EXT_ID:"31501885" OR EXT_ID:"35685304") AND SRC:MED` (2+ ids, still quoted) also works
        fine -- an EPMC/Lucene-side parser quirk specific to a single quoted clause inside parens
        followed by `AND`, not something in this client's control. PMIDs are always purely
        numeric, so quoting was never actually necessary for this field -- dropping the quotes
        fixes both the single-id and multi-id cases uniformly (confirmed live:
        `(EXT_ID:31501885) AND SRC:MED` -> 1 hit). This matters in practice for exactly the kind
        of small hand-picked id list Step 19c's PMID seed file supplies -- a short list very
        commonly lands its last (or only) batch chunk at exactly one id, which silently dropped
        every such record before this fix, with no error raised (`get_by_ids` never raises on a
        partial miss, so this failure mode was invisible). `pmcid`/`doi` lookups are unaffected --
        they never append `AND SRC:MED` in the first place, so the quoted-single-clause case they
        build never hits this specific combination."""
        field = {"pmcid": "PMCID", "pmid": "EXT_ID", "doi": "DOI"}[id_type]
        found: dict[str, dict] = {}
        pbar = tqdm(
            total=len(ids), desc=f"get_by_ids[{id_type}]", unit="id", disable=not show_progress
        )
        try:
            for i in range(0, len(ids), _BATCH_CHUNK_SIZE):
                chunk = ids[i : i + _BATCH_CHUNK_SIZE]
                if id_type == "pmid":
                    clauses = " OR ".join(f"{field}:{value}" for value in chunk)
                    query = f"({clauses}) AND SRC:MED"
                else:
                    clauses = " OR ".join(f'{field}:"{value}"' for value in chunk)
                    query = clauses
                for result in self.search(query, result_type="core", page_size=len(chunk)):
                    key = self._extract_key(result, id_type)
                    if key:
                        found[key] = result
                pbar.update(len(chunk))
        finally:
            pbar.close()
        return found

    @staticmethod
    def _extract_key(result: dict, id_type: Literal["pmcid", "pmid", "doi"]) -> Optional[str]:
        if id_type == "pmcid":
            return result.get("pmcid")
        if id_type == "pmid":
            return result.get("pmid")
        return result.get("doi")

    def close(self) -> None:
        self.session.close()
