"""Loads per-document field values into moros as an allowlisted, reversible `$set`.

This is the partial-update half of the loading story. Its counterpart is `load_documents.py`,
which upserts whole *new* documents through `mongoimport`. The split is deliberate:

- **New documents** -> `mongoimport --mode upsert --upsertFields _id`. That is what both repos'
  roadmaps specify, and it is the right tool for a JSONL of complete documents.
- **Fields on documents that already exist** -> here. A whole-document `$set` would blank whatever
  group it did not mention; the sibling roadmap warns about exactly that ("a classification
  refresh must not blank an `llm_enrichment` group a later enrichment run has populated"). So this
  writes named leaf paths only, checked against `moros_write.WRITE_MODES`.

Currently one mode. Adding another means adding its allowlist to `WRITE_MODES` and its row mapper
below -- both in a diff someone reads, which is the point.

`citations` reads `output/pid_citations.csv` (produced by `join_citations.py`, already keyed on the
document `_id`) and writes `publication_metadata.citation_count` / `.citation_count_updated` /
`.citation_source`. It cannot reach `decision_provenance` or any classification field: refreshing a
number must not be able to relabel who decided the record.

    python3 load_fields.py --mode citations                      # dry run
    python3 load_fields.py --mode citations --limit 100 --confirm # real trial
    python3 load_fields.py --mode citations --confirm             # full load

To undo any run:
    python3 moros_write.py --rollback ../output/rollback/<run_id>.jsonl --confirm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterator

from moros_client import Moros
from moros_write import SafeWriter, new_run_id

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent

DEFAULT_INPUTS = {
    "citations": FOLDER_DIR / "output" / "pid_citations.csv",
    "licences": FOLDER_DIR / "output" / "pid_licences.csv",
}


def citation_row_to_update(row: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
    """One `pid_citations.csv` row -> `(_id, {leaf_path: value})`.

    `citation_count` is written as an **int**, not the CSV's string: the API sorts on this field
    and `"9" > "10"` lexically. A blank count yields no update at all rather than an explicit
    null -- the document already reads null, and writing null over null would put a misleading
    `citation_count_updated` timestamp on a record whose count was never available.
    """
    pid = (row.get("pid") or "").strip()
    raw = (row.get("citation_count") or "").strip()
    if not pid or not raw:
        return None
    return pid, {
        "publication_metadata.citation_count": int(float(raw)),
        "publication_metadata.citation_count_updated": (row.get("citation_count_updated") or "").strip() or None,
        "publication_metadata.citation_source": (row.get("citation_source") or "").strip() or None,
    }


_TRUE = {"y", "yes", "true"}
_FALSE = {"n", "no", "false"}


def licence_row_to_update(row: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
    """One `pid_licences.csv` row -> `(_id, {leaf_path: value})`.

    Writes **two** fields, and the second one is not an oversight.
    `mongo_landscape_export/scripts/schema.py::_resolve_open_access` already fixes the rule for
    this data: EPMC's freshly-fetched flag wins wherever a real lookup happened, falling back to
    the pipeline's original value only when there was none. A licence backfill *is* that lookup,
    so writing the licence while leaving `open_access` on the stale pipeline value would put the
    document in a state the schema says cannot exist.

    An empty `license` is written as `""`, deliberately. It means "EPMC was asked and disclosed no
    licence", which the schema distinguishes from `null`, "never looked up". Skipping it would
    leave the document null and every future backfill would fetch it again forever.

    A row with no licence column at all (a citations-only file) yields no update rather than
    blanking a licence that is already there.
    """
    pid = (row.get("pid") or "").strip()
    if not pid or "license" not in row:
        return None
    update: dict[str, Any] = {"source.access.license": (row.get("license") or "").strip()}
    flag = (row.get("epmc_is_open_access") or "").strip().lower()
    if flag in _TRUE:
        update["source.access.open_access"] = True
    elif flag in _FALSE:
        update["source.access.open_access"] = False
    # EPMC returning no flag leaves open_access alone -- the existing value is the better guess.
    return pid, update


ROW_MAPPERS = {"citations": citation_row_to_update, "licences": licence_row_to_update}


def iter_updates(path: Path, mode: str, limit: int | None) -> Iterator[tuple[str, dict[str, Any]]]:
    mapper = ROW_MAPPERS[mode]
    n = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            update = mapper(row)
            if update is None:
                continue
            yield update
            n += 1
            if limit is not None and n >= limit:
                return


def count_rows(path: Path, mode: str, limit: int | None) -> int:
    return sum(1 for _ in iter_updates(path, mode, limit))


def run(mode: str, input_path: Path, confirm: bool, limit: int | None) -> None:
    if mode not in ROW_MAPPERS:
        raise SystemExit(f"unknown mode {mode!r} -- known: {sorted(ROW_MAPPERS)}")
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist -- run join_citations.py first")

    run_id = new_run_id(f"load_{mode}")
    total = count_rows(input_path, mode, limit)
    print(f"[{run_id}] {input_path.name}: {total:,} documents to update"
          + (f" (--limit {limit})" if limit else ""))

    with Moros.from_env() as moros:
        writer = SafeWriter(moros, mode=mode, run_id=run_id, dry_run=not confirm)
        before = _coverage(moros, mode)
        flips = _open_access_flips(moros, input_path, limit) if mode == "licences" else None
        if flips is not None:
            pct = (flips["flips"] / flips["checked"] * 100) if flips["checked"] else 0.0
            print(f"[{run_id}] open_access: {flips['flips']:,} of {flips['checked']:,} would "
                  f"change ({pct:.2f}%) -- {flips['to_true']:,} to true, "
                  f"{flips['to_false']:,} to false")
            if pct > 5.0:
                print(f"[{run_id}] !! that is far above the 0.05% seen when this rule was first "
                      f"applied. Worth understanding before writing.")
        result = writer.apply_streaming(
            iter_updates(input_path, mode, limit), total=total, desc=f"load[{mode}]"
        )
        after = _coverage(moros, mode) if confirm else before
        if confirm:
            print(f"[{run_id}] coverage: {before:,} -> {after:,} documents "
                  f"({after / moros.count() * 100:.2f}% of the corpus)")
        writer.write_report({
            "input": str(input_path),
            "rows": total,
            "limit": limit,
            "result": result.as_dict(),
            "coverage_before": before,
            "coverage_after": after,
            "open_access_flips": flips,
        })
        if confirm and result.errors:
            raise SystemExit(f"[{run_id}] finished with {len(result.errors)} batch error(s) -- "
                             f"see the report; re-running is safe and skips nothing")


def _open_access_flips(moros: Moros, input_path: Path, limit: int | None) -> dict[str, int]:
    """How many `open_access` values this load would actually change, reported BEFORE writing.

    The licence backfill writes two fields, and the second one has a wider blast radius than the
    first: `license` is filling a null, but `open_access` already holds a value on every document.
    The established rule says EPMC's fresh flag wins -- and when that rule was first applied it
    moved 395 of 738,198 rows, 0.05%. A materially larger number here means EPMC and this corpus
    disagree about far more than they did, which is a reason to stop and look rather than a
    detail to note afterwards."""
    incoming: dict[str, bool] = {}
    for pid, update in iter_updates(input_path, "licences", limit):
        if "source.access.open_access" in update:
            incoming[pid] = update["source.access.open_access"]
    if not incoming:
        return {"checked": 0, "flips": 0, "to_true": 0, "to_false": 0}

    ids = list(incoming)
    stats = {"checked": len(ids), "flips": 0, "to_true": 0, "to_false": 0}
    for start in range(0, len(ids), 5_000):
        chunk = ids[start:start + 5_000]
        for doc in moros.collection.find(
            {"_id": {"$in": chunk}}, {"source.access.open_access": 1}
        ):
            current = ((doc.get("source") or {}).get("access") or {}).get("open_access")
            new = incoming[doc["_id"]]
            if current is not None and current != new:
                stats["flips"] += 1
                stats["to_true" if new else "to_false"] += 1
    return stats


def _coverage(moros: Moros, mode: str) -> int:
    """How many documents already carry this mode's headline field. Reported before and after so a
    load's real effect is a number, not an assumption."""
    field = {"citations": "publication_metadata.citation_count",
             "licences": "source.access.license"}[mode]
    return moros.count({field: {"$ne": None}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(ROW_MAPPERS))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Update only the first N documents -- a real, reversible trial.")
    args = parser.parse_args()
    run(args.mode, args.input or DEFAULT_INPUTS[args.mode], args.confirm, args.limit)


if __name__ == "__main__":
    main()
