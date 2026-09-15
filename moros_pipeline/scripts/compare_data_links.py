"""Compares two builds of `pid_data_links.csv` before a load: per resource, the documents that gain or
lose it and the links added or removed; every link that moved to another resource or a canonical id
(an ArrayExpress `E-GEOD` becoming a GEO series, a versioned dbGaP study becoming the study, a PXD
re-homed to the partner hosting it); and the documents newly withheld. The v1.5.0 rebuild must
change the corpus only in the ways `docs/data_links_sources.md` says; this checks it rather than
assuming it.

A link missing from `links[]` whose resource is still present with at least the same `count` was
capped out of the stored detail (50 per resource, 300 per document), not lost; it is reported apart.
So is a link whose canonical id the new build already holds under another resource: the two were
one accession, and the dedupe merged them (an `E-GEOD` mirror of a GEO series text-mined as both).

Reads the two files row by row (a build writes them in the same pid order); pids out of step are
matched from a buffer. Writes a JSON report beside them.

    python3 compare_data_links.py --old ../output/pid_data_links.v140.csv --new ../output/pid_data_links.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from ebisearch_resources import canonicalise

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR.parent / "output"
EPMC_ROUTES = frozenset({"tm_accession", "tm_supplementary", "ext_links", "derived"})
MAX_EXAMPLES = 5


def _block(cell: str | None) -> dict | None:
    cell = (cell or "").strip()
    return json.loads(cell) if cell else None


def _links(block: dict | None) -> set[tuple[str, str]]:
    return {(link["resource"], str(link["id"]).lower()) for link in (block or {}).get("links") or []}


def _counts(block: dict | None) -> dict[str, int]:
    return {r["resource"]: int(r.get("count") or 0) for r in (block or {}).get("resources") or []}


class Comparison:
    def __init__(self) -> None:
        self.documents: Counter = Counter()
        self.gained_documents: Counter = Counter()   # resource -> documents that gain it
        self.lost_documents: Counter = Counter()     # resource -> documents that lose it
        self.added_links: Counter = Counter()
        self.removed_links: Counter = Counter()      # really gone: not moved, merged or capped
        self.merged_links: Counter = Counter()       # resource -> links merged into one held elsewhere
        self.removed_under_cap: Counter = Counter()
        self.moves: Counter = Counter()              # (old resource, new resource) -> links
        self.confirmed_by_both: Counter = Counter()  # resource -> documents where both routes found it
        self.examples: dict[str, list[str]] = {}

    def _example(self, kind: str, pid: str, text: str) -> None:
        bucket = self.examples.setdefault(kind, [])
        if len(bucket) < MAX_EXAMPLES:
            bucket.append(f"{pid}: {text}")

    def add(self, pid: str, old: dict | None, new: dict | None) -> None:
        self.documents["compared"] += 1
        if old is None and new is None:
            self.documents["withheld_in_both"] += 1
            return
        if new is None:
            self.documents["newly_withheld"] += 1
            self._example("newly_withheld", pid, "resolved before, withheld now")
            return
        if old is None:
            self.documents["newly_resolved"] += 1
        if "ebisearch" in (new.get("sources") or []):
            self.documents["with_ebisearch"] += 1

        old_links, new_links = _links(old), _links(new)
        old_counts, new_counts = _counts(old), _counts(new)
        for resource in new_counts.keys() - old_counts.keys():
            self.gained_documents[resource] += 1
        for resource in old_counts.keys() - new_counts.keys():
            self.lost_documents[resource] += 1

        added = new_links - old_links
        added_by_id = {link_id: resource for resource, link_id in added}
        new_ids = {link_id for _, link_id in new_links}
        matched: set[tuple[str, str]] = set()
        for resource, link_id in sorted(old_links - new_links):
            canonical = canonicalise(resource, link_id)[1].lower()
            target = next((i for i in (link_id, canonical) if i in added_by_id), None)
            if target is not None:
                self.moves[(resource, added_by_id[target])] += 1
                matched.add((added_by_id[target], target))
                self._example("moved", pid, f"{resource}:{link_id} -> {added_by_id[target]}:{target}")
            elif link_id in new_ids or canonical in new_ids:
                self.merged_links[resource] += 1
                self._example("merged", pid, f"{resource}:{link_id} into a link already held")
            elif resource in new_counts and new_counts[resource] >= old_counts.get(resource, 0):
                self.removed_under_cap[resource] += 1
            else:
                self.removed_links[resource] += 1
                self._example("removed", pid, f"{resource}:{link_id}")
        for resource, _ in added - matched:
            self.added_links[resource] += 1

        for entry in new.get("resources") or []:
            routes = set(entry.get("routes") or [])
            if routes & EPMC_ROUTES and any(r.startswith("ebisearch") for r in routes):
                self.confirmed_by_both[entry["resource"]] += 1

    def as_dict(self) -> dict:
        def table(counter: Counter) -> dict:
            return {(" -> ".join(k) if isinstance(k, tuple) else k): v for k, v in counter.most_common()}
        return {"documents": dict(self.documents), "gained_documents": table(self.gained_documents),
                "lost_documents": table(self.lost_documents), "added_links": table(self.added_links),
                "removed_links": table(self.removed_links), "merged_links": table(self.merged_links),
                "removed_under_cap": table(self.removed_under_cap), "moves": table(self.moves),
                "confirmed_by_both": table(self.confirmed_by_both), "examples": self.examples}


def compare(old_path: Path, new_path: Path) -> Comparison:
    result = Comparison()
    pending_old: dict[str, str] = {}
    pending_new: dict[str, str] = {}
    with old_path.open(newline="", encoding="utf-8") as fo, new_path.open(newline="", encoding="utf-8") as fn:
        for old_row, new_row in zip_longest(csv.DictReader(fo), csv.DictReader(fn)):
            if old_row and new_row and old_row["pid"] == new_row["pid"]:
                result.add(old_row["pid"], _block(old_row["data_links_json"]),
                           _block(new_row["data_links_json"]))
                continue
            for row, mine, theirs, is_old in ((old_row, pending_old, pending_new, True),
                                              (new_row, pending_new, pending_old, False)):
                if row is None:
                    continue
                pid = row["pid"]
                if pid in theirs:
                    other = theirs.pop(pid)
                    old_cell, new_cell = (row["data_links_json"], other) if is_old else (other, row["data_links_json"])
                    result.add(pid, _block(old_cell), _block(new_cell))
                else:
                    mine[pid] = row["data_links_json"]
    result.documents["only_in_old"] = len(pending_old)
    result.documents["only_in_new"] = len(pending_new)
    return result


def render(result: Comparison) -> None:
    d = result.documents
    print(f"compare_data_links: {d['compared']:,} documents compared; {d['newly_withheld']:,} newly "
          f"withheld, {d['newly_resolved']:,} newly resolved, {d['withheld_in_both']:,} withheld in both; "
          f"{d['with_ebisearch']:,} with EBI Search; {d['only_in_old']:,} only in old, "
          f"{d['only_in_new']:,} only in new")
    for title, counter in (("documents gaining a resource", result.gained_documents),
                           ("documents losing a resource", result.lost_documents),
                           ("links moved (old -> new resource)", result.moves),
                           ("links added", result.added_links),
                           ("links merged into one already held (same accession)", result.merged_links),
                           ("links removed (not moved, merged or capped)", result.removed_links),
                           ("links capped out of the stored detail", result.removed_under_cap),
                           ("documents where Europe PMC and EBI Search both found a resource",
                            result.confirmed_by_both)):
        if counter:
            print(f"\n  {title}:")
            for key, n in counter.most_common(25):
                label = " -> ".join(key) if isinstance(key, tuple) else key
                print(f"    {label:<36} {n:>9,}")
    for kind, lines in result.examples.items():
        print(f"\n  e.g. {kind}: " + "; ".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    render(result)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report = args.report or OUTPUT_DIR / f"compare_data_links_{stamp}.report.json"
    report.write_text(json.dumps({"old": str(args.old), "new": str(args.new), **result.as_dict()},
                                 indent=1) + "\n", encoding="utf-8")
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()
