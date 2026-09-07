"""Shared lookup over `epmc_citations.csv`, used by both the staging chain and the moros loader.

`fetch_citations.py` fetches each record under *one* key (pmid if it has one, else doi, else
pmcid) so nothing is fetched twice. Joining is the opposite problem: given a record, find its
count under **whichever** key it was fetched by. Trying all three is both simpler than
re-deriving the fetch-time assignment and strictly better -- a record whose pmid EPMC did not
return can still match on a doi that was fetched for it.

Lives here, next to `pid.py`, because it is an identifier-priority rule about this project's own
records, and because `moros_pipeline/` already depends on this folder (for `pid.py` and
`schema.py`) while nothing here depends on `moros_pipeline/`. One direction only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# The order a record's identifiers are tried. Not the same as the fetch-time assignment order:
# there, priority avoids duplicate work; here, it is only a tie-break between two keys that both
# resolved, which in practice describe the same paper.
LOOKUP_ORDER = ("pmid", "doi", "pmcid")


def load_citation_index(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(key_type, key) -> (citation_count, fetched_at). A later row supersedes an earlier one for
    the same key, so a refresh appended to the same file wins over the original fetch.

    Rows where EPMC returned the record but no count are skipped entirely rather than stored as
    an empty string -- "no count available" and "count is zero" must not collapse.
    """
    index: dict[tuple[str, str], tuple[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            count = (row.get("citation_count") or "").strip()
            if not count:
                continue
            index[(row["key_type"], row["key"])] = (count, (row.get("fetched_at") or "").strip())
    return index


def lookup_citation(
    index: dict[tuple[str, str], tuple[str, str]],
    pmid: str = "",
    pmcid: str = "",
    doi: str = "",
) -> tuple[str, str, str] | None:
    """Returns `(citation_count, fetched_at, matched_key_type)` or None. DOIs are lowercased to
    match how `fetch_citations.py` normalises its keys."""
    values = {"pmid": (pmid or "").strip(),
              "pmcid": (pmcid or "").strip(),
              "doi": (doi or "").strip().lower()}
    for key_type in LOOKUP_ORDER:
        value = values[key_type]
        if not value:
            continue
        hit = index.get((key_type, value))
        if hit is not None:
            return hit[0], hit[1], key_type
    return None
