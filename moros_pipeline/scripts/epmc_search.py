"""Minimal Europe PMC search client: cursorMark deep pagination, retrying session, tqdm.

A deliberate, documented mirror of `src/dome_triage/ingest/epmc_client.py::EpmcClient.search`,
for the same reason `mongo_landscape_export/scripts/schema.py` mirrors `TIER_MODEL_IDS`: this
folder runs on the host with plain `python3` and cannot import the `dome_triage` package, which
lives only inside the Docker image. The paging contract (cursorMark, stop when `nextCursorMark`
repeats or results are empty) is reproduced exactly, so a window fetched here is identical to one
fetched by the pipeline's own client.

If the upstream client's paging behaviour ever changes, change it here too.
"""

from __future__ import annotations

from typing import Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def create_session(max_retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries, backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class EpmcSearch:
    def __init__(self, page_size: int = 1000) -> None:
        self.page_size = page_size
        self.session = create_session()

    def count(self, query: str) -> int:
        """One request, `pageSize=1`, `resultType=idlist` -- reads the API's own hitCount without
        pulling any metadata."""
        resp = self.session.get(
            f"{BASE_URL}/search",
            params={"query": query, "pageSize": 1, "format": "json", "resultType": "idlist"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("hitCount", 0)

    def search(
        self, query: str, result_type: str = "core", show_progress: bool = True,
        page_size: Optional[int] = None,
    ) -> Iterator[dict]:
        cursor = "*"
        size = page_size or self.page_size
        bar = tqdm(total=None, unit="rec", desc=query[:44], disable=not show_progress)
        try:
            while True:
                resp = self.session.get(
                    f"{BASE_URL}/search",
                    params={"query": query, "pageSize": size, "cursorMark": cursor,
                            "format": "json", "resultType": result_type},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                if bar.total is None:
                    bar.total = data.get("hitCount", 0)
                results = data.get("resultList", {}).get("result", [])
                if not results:
                    return
                yield from results
                bar.update(len(results))
                next_cursor = data.get("nextCursorMark", "")
                if not next_cursor or next_cursor == cursor:
                    return
                cursor = next_cursor
        finally:
            bar.close()

    def close(self) -> None:
        self.session.close()
