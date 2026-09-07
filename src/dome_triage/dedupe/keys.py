"""Union-find clustering: two RawRecords merge into the same cluster if they share ANY normalized
ID (pmcid, doi, or pmid) between them -- not just a single priority field. This matters because
Adapter B rows (configs/sources.yaml `id_pair_only`) carry no DOI, so matching only on "the first
non-null field in priority order" would miss real duplicates that share a DOI or PMID with a row
from another source. The configured `dedup.id_priority` in sources.yaml is used only to choose the
*displayed* canonical_key for an already-formed cluster (see choose_canonical_key below), never to
decide cluster membership.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from dome_triage.schema import RawRecord

ID_FIELDS = ("pmcid", "doi", "pmid")
_KEY_PREFIX = {"pmcid": "PMCID", "doi": "DOI", "pmid": "PMID"}


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _record_ids(record: RawRecord) -> list[str]:
    return [
        f"{field}:{value}"
        for field in ID_FIELDS
        if (value := getattr(record, field))
    ]


def build_clusters(records: list[RawRecord]) -> list[list[RawRecord]]:
    """Group records sharing any normalized ID into clusters. Records with no usable ID
    (shouldn't occur here -- source_loaders routes ID-less rows to `unresolved` instead) each
    become their own singleton cluster so no data is silently dropped."""
    uf = UnionFind()
    for record in records:
        ids = _record_ids(record)
        first, rest = (ids[0], ids[1:]) if ids else (None, [])
        for other_id in rest:
            uf.union(first, other_id)

    clusters_by_root: dict[str, list[RawRecord]] = {}
    standalone: list[list[RawRecord]] = []
    for record in records:
        ids = _record_ids(record)
        if not ids:
            standalone.append([record])
            continue
        root = uf.find(ids[0])
        clusters_by_root.setdefault(root, []).append(record)

    return list(clusters_by_root.values()) + standalone


def choose_canonical_key(
    cluster: list[RawRecord], priority: tuple[str, ...] = ("pmcid", "doi", "pmid")
) -> str:
    """Pick the displayed key for an already-formed cluster, using the configured priority order."""
    for field in priority:
        for record in cluster:
            value = getattr(record, field)
            if value:
                return f"{_KEY_PREFIX[field]}:{value}"
    raise ValueError("Cluster has no usable ID on any contributing record")


# Values that look present to a bare `if value:` check but are not real identifiers. This has bitten
# this project three separate times (2026-08-27), each time collapsing many genuinely distinct
# records onto a single shared record_id:
#   1. `pd.read_csv(dtype=str)` leaves a missing id as float('nan'), which is TRUTHY in Python, so
#      `if value:` passed it straight through and produced the key "PMCID:nan".
#   2. `Series.astype(str)` / any str() round-trip turns that same NaN into the literal string
#      "nan", which is also truthy and produces the identical key.
#   3. A 1,000-record Step 23a test run classified 250 genuinely different papers that had all
#      collapsed onto that one "PMCID:nan" hash -- confirmed by their differing classifications.
# Callers should still normalize NaN -> None upstream; this is the last line of defence so a
# missed conversion degrades to "no id" (correct, records stay distinct) instead of silently
# fusing unrelated records together.
_NON_ID_STRINGS = {"nan", "none", "null", "na", "n/a", "<na>", "nat", ""}


def _is_usable_id(value: object) -> bool:
    if value is None:
        return False
    # float('nan') != itself -- catches a NaN that never went through a str() conversion.
    if isinstance(value, float) and value != value:
        return False
    return str(value).strip().lower() not in _NON_ID_STRINGS


def canonical_key_from_ids(
    pmcid: Optional[str],
    pmid: Optional[str],
    doi: Optional[str],
    priority: tuple[str, ...] = ("pmcid", "doi", "pmid"),
) -> Optional[str]:
    """Same priority-order key construction as `choose_canonical_key`, but from three raw scalar
    ID values directly rather than a formed RawRecord cluster -- for callers that only have a
    single record's ids on hand (e.g. a bulk-match candidate row that has never been through
    `dedupe consolidate`) and need the *same* key a future consolidate run would produce for it.
    Returns None (never raises) when none of the three ids is present -- a valid, expected
    outcome for a standalone caller, unlike `choose_canonical_key`'s cluster context where every
    record is guaranteed at least one id by construction (source_loaders routes id-less rows to
    `unresolved` instead)."""
    values = {"pmcid": pmcid, "doi": doi, "pmid": pmid}
    for field in priority:
        value = values[field]
        if _is_usable_id(value):
            return f"{_KEY_PREFIX[field]}:{str(value).strip()}"
    return None


def record_id_from_canonical_key(canonical_key: str) -> str:
    """The one hash `consolidate.py`'s `_merge_cluster` and any other caller must use to turn a
    canonical_key into the `record_id` stamped on a CanonicalRecord -- factored out here so a
    caller building a record *outside* the consolidate pipeline (e.g. curating directly from the
    bulk pool before it's been through `dedupe consolidate`) computes the exact same id a later
    consolidate run would, letting the two reconcile without an explicit join."""
    return hashlib.sha1(canonical_key.encode()).hexdigest()


def record_id_from_ids(
    pmcid: Optional[str],
    pmid: Optional[str],
    doi: Optional[str],
    priority: tuple[str, ...] = ("pmcid", "doi", "pmid"),
) -> Optional[str]:
    """Convenience composition of the two functions above -- the single call most callers with
    raw scalar ids actually want. Returns None when no id is present at all."""
    canonical_key = canonical_key_from_ids(pmcid, pmid, doi, priority)
    return record_id_from_canonical_key(canonical_key) if canonical_key else None
