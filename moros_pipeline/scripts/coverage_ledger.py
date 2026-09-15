"""Which (query, time window) pairs have been fetched, and which have been loaded into moros.

This is what makes an incremental run answer "what is genuinely not covered yet?" instead of
re-fetching everything and relying on dedupe to throw the duplicates away.

Two authorities, deliberately kept separate rather than merged:

- **The ledger** is authoritative for *what was fetched* -- it is the only record that a window
  was ever asked for, including windows that legitimately returned nothing.
- **moros** is authoritative for *what was loaded*.

They can legitimately differ (a window fetched but not yet classified and loaded), but they must
differ in only one direction. Documents in moros for a window the ledger has never seen means the
ledger has lost history, and `fetch_search_space.py` stops rather than guessing -- because the
failure mode of guessing is exactly the one this whole finalisation exists to fix: quietly
deciding a population is already handled when it is not.

Coverage is keyed on `sha256(canonical query string)`, so editing `config/search_space.yaml`
starts a new key rather than inheriting the previous query's windows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_CONFIG = FOLDER_DIR / "config" / "search_space.yaml"
DEFAULT_LEDGER = FOLDER_DIR / "output" / "coverage_ledger.json"

# Europe PMC's first-index date, which an index window searches on (see SearchSpace.index_query), and
# the far end of the publication-date range for papers dated in the future.
INDEX_DATE_FIELD = "FIRST_IDATE"
FUTURE_DATED_UNTIL = "2100-12-31"


@dataclass
class SearchSpace:
    name: str
    terms: list[str]
    combine: str
    sources: list[str]
    date_field: str
    coverage_start: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "SearchSpace":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not data.get("terms"):
            raise ValueError(f"{path} defines no terms -- refusing to build an empty query")
        return cls(
            name=data.get("name", "unnamed"),
            terms=list(data["terms"]),
            combine=data.get("combine", "OR").upper(),
            sources=list(data.get("sources") or []),
            date_field=data.get("date_field", "FIRST_PDATE"),
            coverage_start=str(data.get("coverage_start", "1900-01-01")),
            raw=data,
        )

    def term_clause(self) -> str:
        return f" {self.combine} ".join(self.terms)

    def query(self, date_from: str, date_to: str) -> str:
        """The exact EPMC query for a window. Mirrors `ingest/bulk_match.py::_range_query`'s shape
        exactly -- `({terms}) AND ({DATE_FIELD}:[from TO to])` -- so windows fetched by the old
        hardcoded path and by this one are directly comparable."""
        clause = f"({self.term_clause()}) AND ({self.date_field}:[{date_from} TO {date_to}])"
        if self.sources:
            clause += " AND (" + " OR ".join(f"SRC:{s}" for s in self.sources) + ")"
        return clause

    def index_query(self, since: str, up_to: str, future_dated_from: str | None = None) -> str:
        """Everything Europe PMC first indexed from `since` to `up_to`, whatever its publication
        date: new papers, papers indexed late for an earlier year, and papers dated in the future.
        With `future_dated_from`, also every paper whose publication date is on or after that day --
        used once, when a chain of index windows starts after a year window, for papers indexed
        before that fetch but dated after its end, which neither kind of window would otherwise
        return. Same terms and sources as `query`, so it belongs to the same search space and hash."""
        date_clause = f"{INDEX_DATE_FIELD}:[{since} TO {up_to}]"
        if future_dated_from:
            date_clause += f" OR {self.date_field}:[{future_dated_from} TO {FUTURE_DATED_UNTIL}]"
        clause = f"({self.term_clause()}) AND ({date_clause})"
        if self.sources:
            clause += " AND (" + " OR ".join(f"SRC:{s}" for s in self.sources) + ")"
        return clause

    def canonical(self) -> str:
        """What the hash is taken over. Sorted and normalised so a cosmetic reordering of the YAML
        does not invalidate real coverage, while any change to what is actually searched does."""
        return json.dumps({
            "terms": sorted(self.terms),
            "combine": self.combine,
            "sources": sorted(self.sources),
            "date_field": self.date_field,
        }, sort_keys=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


class CoverageLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists() else {"queries": []}
        )

    def entry(self, space: SearchSpace, create: bool = False) -> dict[str, Any] | None:
        digest = space.sha256()
        for item in self.data["queries"]:
            if item["query_sha256"] == digest:
                return item
        if not create:
            return None
        item = {
            "query_sha256": digest,
            "name": space.name,
            "canonical": space.canonical(),
            "example_query": space.query(space.coverage_start, "2026-01-01"),
            "windows": [],
        }
        self.data["queries"].append(item)
        return item

    def covered_windows(self, space: SearchSpace) -> list[tuple[str, str]]:
        item = self.entry(space)
        if item is None:
            return []
        return [(w["from"], w["to"]) for w in item["windows"]]

    def covered_years(self, space: SearchSpace) -> set[int]:
        """Every year some window touches -- what the cross-check against moros asks about."""
        years: set[int] = set()
        for start, end in self.covered_windows(space):
            years.update(range(int(start[:4]), int(end[:4]) + 1))
        return years

    def complete_years(self, space: SearchSpace) -> set[int]:
        """Years some window covers whole, 1 January to 31 December. A refresh run with --up-to
        before 31 December leaves a window that stops partway through its year; counting that year
        as done would leave the rest of it unfetched once the calendar turns, because only the
        current year is re-fetched."""
        years: set[int] = set()
        for start, end in self.covered_windows(space):
            first = int(start[:4]) + (0 if start[5:] == "01-01" else 1)
            last = int(end[:4]) - (0 if end[5:] == "12-31" else 1)
            years.update(range(first, last + 1))
        return years

    def record_window(
        self, space: SearchSpace, date_from: str, date_to: str, fetched: int, path: str
    ) -> None:
        item = self.entry(space, create=True)
        assert item is not None
        for window in item["windows"]:
            if window["from"] == date_from and window["to"] == date_to:
                window.update({"fetched": fetched, "fetched_at": _now(), "path": path})
                return
        item["windows"].append({
            "from": date_from, "to": date_to, "fetched": fetched,
            "fetched_at": _now(), "loaded_to_moros": None, "loaded_at": None, "path": path,
        })

    def index_windows(self, space: SearchSpace) -> list[dict[str, Any]]:
        item = self.entry(space)
        return list(item.get("index_windows", [])) if item else []

    def record_index_window(
        self, space: SearchSpace, since: str, up_to: str, fetched: int, path: str,
        future_dated_from: str | None = None,
    ) -> None:
        item = self.entry(space, create=True)
        assert item is not None
        windows = item.setdefault("index_windows", [])
        fields = {"fetched": fetched, "fetched_at": _now(), "path": path,
                  "future_dated_from": future_dated_from}
        for window in windows:
            if window["from"] == since and window["to"] == up_to:
                window.update(fields)
                return
        windows.append({"from": since, "to": up_to, **fields})

    def indexed_through(self, space: SearchSpace) -> str | None:
        """The last day up to which every record Europe PMC had indexed has been fetched. A year
        window returns what was indexed by the day it ran; an index window, what was indexed by its
        `to` or by the day it ran, whichever is earlier. None before any fetch."""
        days = [w["fetched_at"][:10] for w in (self.entry(space) or {}).get("windows", [])
                if w.get("fetched_at")]
        days += [min(w["to"], w["fetched_at"][:10]) for w in self.index_windows(space)
                 if w.get("fetched_at")]
        return max(days) if days else None

    def future_dated_from(self, space: SearchSpace) -> str | None:
        """Before the first index window: the day after the end of the most recently fetched year
        window. Papers indexed before that fetch but dated after its end are in no window yet.
        Once an index window exists it has already fetched them, so None."""
        if self.index_windows(space):
            return None
        windows = [w for w in (self.entry(space) or {}).get("windows", []) if w.get("fetched_at")]
        if not windows:
            return None
        latest = max(windows, key=lambda w: (w["fetched_at"], w["to"]))
        return (date.fromisoformat(latest["to"]) + timedelta(days=1)).isoformat()

    def record_loaded(self, space: SearchSpace, date_from: str, date_to: str, loaded: int) -> None:
        item = self.entry(space, create=True)
        assert item is not None
        for window in item["windows"]:
            if window["from"] == date_from and window["to"] == date_to:
                window["loaded_to_moros"] = loaded
                window["loaded_at"] = _now()
                return
        raise KeyError(f"no fetched window {date_from}..{date_to} to mark as loaded")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def missing_years(space: SearchSpace, ledger: CoverageLedger, up_to: date) -> list[int]:
    """The years this query has never been fetched for, from `coverage_start` to `up_to`. Whole
    years, matching the per-year checkpointing `bulk_match.py` already uses -- except the current
    year, which is always re-fetched because it is still filling up."""
    first = int(space.coverage_start[:4])
    covered = ledger.complete_years(space)
    years = [y for y in range(first, up_to.year + 1) if y not in covered]
    if up_to.year not in years:
        years.append(up_to.year)  # the current year is never "done"
    return sorted(set(years))


def plan_index_window(space: SearchSpace, ledger: CoverageLedger, up_to: date,
                      indexed_since: str) -> tuple[str, str | None]:
    """(since, future_dated_from) for an index-date fetch, or SystemExit when it would leave a gap.

    `indexed_since` is a date or "last" (the ledger's `indexed_through`). A since later than that
    day would skip whatever was indexed in between, so it is refused; an earlier one only overlaps,
    and the moros dedupe drops what is already loaded. Every year before `up_to` must have been
    fetched at least once: an index window only adds to coverage that exists."""
    through = ledger.indexed_through(space)
    if through is None:
        raise SystemExit("nothing has been fetched for this query yet -- run the year windows first "
                         "(fetch_search_space.py --up-to today)")
    first = int(space.coverage_start[:4])
    never = [y for y in range(first, up_to.year) if y not in ledger.covered_years(space)]
    if never:
        raise SystemExit(f"years never fetched for this query: {never[:10]}"
                         f"{' ...' if len(never) > 10 else ''} -- run without --indexed-since first")
    since = through if indexed_since == "last" else date.fromisoformat(indexed_since).isoformat()
    if since > through:
        raise SystemExit(f"--indexed-since {since} starts after {through}, the last day already "
                         f"covered: records indexed in between would never be fetched. Use 'last' or "
                         f"an earlier date.")
    if since > up_to.isoformat():
        raise SystemExit(f"nothing to fetch: everything Europe PMC indexed is already covered up to "
                         f"{since}, past --up-to {up_to.isoformat()}")
    return since, ledger.future_dated_from(space)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
