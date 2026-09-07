"""The only module in this folder that opens a connection to moros.

Read-only by default and by shape: this module exposes counts, histograms, index listings and
document lookups, and nothing that writes. Writes go through `moros_write.py`, which takes the
collection handle from here but wraps it in an allowlist and a rollback snapshot.

Why a separate module at all: `observatory-ws` in the sibling repo is read-only by design and
"must never gain write credentials", and the same discipline is what makes this folder safe to
point at production. Keeping every connection in one file means there is exactly one place to
audit, and exactly one place that reads the internal host address out of `.env`.

Server constraints that shape everything here, measured against the real server:
- MongoDB **4.2.25**, standalone (not a replica set). `directConnection` is implied by a single
  host with no replicaSet option; no transactions, no `allowDiskUse` on `find()`.
- No authentication. The URI carries no credentials -- the address itself is the only sensitive
  part, which is why `.env` is gitignored and only `.env.example` is tracked.
- moros hosts other production databases besides this one. Every query here is bounded by
  `maxTimeMS`, and no query fans out over the whole 1.5GB collection without saying so.

Usage:
    from moros_client import Moros
    with Moros.from_env() as moros:
        print(moros.count())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pymongo import MongoClient

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_ENV_PATH = FOLDER_DIR / ".env"

# NOTE two spellings, both correct in their place: `find`/`find_one` build a Cursor and take
# `max_time_ms`, while `count_documents`/`aggregate` are commands and take `maxTimeMS`.
# 5s matches observatory-ws's MONGO_MAX_TIME_MS; the long budget is for the deliberate
# whole-collection aggregations (a provenance histogram over 827k documents takes seconds).
DEFAULT_MAX_TIME_MS = 5_000
LONG_MAX_TIME_MS = 120_000

# `_id` is a UUID5 **string**, not an ObjectId. Getting this wrong is the classic failure on this
# collection: every lookup silently returns nothing rather than erroring.
ID_FIELD = "_id"


def load_env(path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Minimal KEY=VALUE reader -- deliberately not python-dotenv, to keep this folder's host
    dependency list at pandas/tqdm/requests/pymongo. Real environment variables win, so a one-off
    run can override without editing the file."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    for key in ("MONGODB_URI", "MONGODB_DB", "MONGODB_COLLECTION"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    missing = [k for k in ("MONGODB_URI", "MONGODB_DB", "MONGODB_COLLECTION") if not values.get(k)]
    if missing:
        raise ValueError(
            f"missing {', '.join(missing)} -- copy {DEFAULT_ENV_PATH.name}.example to "
            f"{DEFAULT_ENV_PATH} and fill in the real values (see .env.example for what they are)"
        )
    return values


class Moros:
    """A read-only view of one collection, plus the handle `moros_write.py` needs."""

    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client: MongoClient = MongoClient(
            uri,
            serverSelectionTimeoutMS=5_000,
            maxPoolSize=10,
        )
        self.collection = self._client[db_name][collection_name]

    @classmethod
    def from_env(cls, env_path: Path = DEFAULT_ENV_PATH) -> "Moros":
        env = load_env(env_path)
        return cls(env["MONGODB_URI"], env["MONGODB_DB"], env["MONGODB_COLLECTION"])

    def __enter__(self) -> "Moros":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- identity / connectivity -------------------------------------------------

    def server_version(self) -> str:
        return self._client.server_info()["version"]

    def describe(self) -> str:
        """One line for the top of every run report -- what we are actually pointed at. The host
        is included because pointing a write at the wrong server is the failure mode that matters
        most, and it should be impossible to miss in a log."""
        host = self.uri.split("://", 1)[-1]
        return (
            f"{host}/{self.db_name}.{self.collection_name} "
            f"(MongoDB {self.server_version()}, {self.count():,} documents)"
        )

    # -- counts and histograms ---------------------------------------------------

    def count(self, query: Optional[dict] = None) -> int:
        return self.collection.count_documents(query or {}, maxTimeMS=LONG_MAX_TIME_MS)

    def histogram(self, field: str) -> dict[Any, int]:
        """`$group` over one field across the whole collection. Seconds, not milliseconds -- only
        used by the verifier and the migration, never on a request path."""
        pipeline = [{"$group": {"_id": f"${field}", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}]
        cursor = self.collection.aggregate(pipeline, maxTimeMS=LONG_MAX_TIME_MS, allowDiskUse=True)
        return {doc["_id"]: doc["n"] for doc in cursor}

    def indexes(self) -> dict[str, dict]:
        """Index name -> definition. `positives_text` and `class_year_id` are the two that matter:
        the backend degrades silently to a regex scan when the text index is absent, so 'search
        still works' is not evidence that it is there."""
        return {name: spec for name, spec in self.collection.index_information().items()}

    # -- lookups -----------------------------------------------------------------

    def find_one(self, query: dict, projection: Optional[dict] = None) -> Optional[dict]:
        return self.collection.find_one(query, projection, max_time_ms=DEFAULT_MAX_TIME_MS)

    def get(self, doc_id: str, projection: Optional[dict] = None) -> Optional[dict]:
        return self.find_one({ID_FIELD: doc_id}, projection)

    def existing_ids(self, ids: Iterable[str], batch_size: int = 5_000) -> set[str]:
        """Which of these `_id`s are already in the collection. Batched `$in` against the `_id`
        index -- this is how an incremental run decides what is genuinely new without pulling a
        827k-element id set to the client."""
        found: set[str] = set()
        batch: list[str] = []
        for doc_id in ids:
            batch.append(doc_id)
            if len(batch) >= batch_size:
                found |= self._existing_batch(batch)
                batch = []
        if batch:
            found |= self._existing_batch(batch)
        return found

    def _existing_batch(self, batch: list[str]) -> set[str]:
        cursor = self.collection.find(
            {ID_FIELD: {"$in": batch}}, {ID_FIELD: 1}, max_time_ms=LONG_MAX_TIME_MS
        )
        return {doc[ID_FIELD] for doc in cursor}

    def iter_documents(
        self, query: dict, projection: dict, batch_size: int = 1_000
    ) -> Iterator[dict]:
        """Keyset-free streaming read. Sorted by `_id` (indexed) so a long export is stable and
        resumable rather than depending on natural order."""
        cursor = (
            self.collection.find(query, projection, no_cursor_timeout=False)
            .sort(ID_FIELD, 1)
            .batch_size(batch_size)
        )
        yield from cursor
