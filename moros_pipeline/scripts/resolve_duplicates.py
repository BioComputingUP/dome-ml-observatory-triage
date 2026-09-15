#!/usr/bin/env python3
"""Removes the second copy of a paper that was loaded twice under two different `_id`s.

Why they exist: the `_id` is UUID5 of the first of pmcid > doi > pmid (`pid.py`). Europe PMC adds a
PMCID to a paper weeks after publication, so a fetch that sees the PMCID and a fetch that does not
mint different ids for the same paper. The 2026-09-03 load compared only `_id` and inserted 10,572
second copies (9,995 of them an August copy with a PMCID beside a September copy without).
`build_incoming_documents.py` has checked pmcid / doi / pmid against the corpus since 2026-09-15, so
no new load can add one; this removes the ones already there.

Read-only by default, like every other writer in this folder:

    python3 resolve_duplicates.py                        # classify every group, write the report
    python3 resolve_duplicates.py --limit 50 --confirm   # a real, restorable 50-group trial
    python3 resolve_duplicates.py --confirm              # the rest
    python3 resolve_duplicates.py --restore ../output/rollback/<run_id>.deleted.jsonl --confirm

**Which copy stays**: the one whose `_id` is what `mint_landscape_pid` makes of the group's
identifiers combined -- the id every future fetch of that paper will mint, so the corpus stops
drifting from Europe PMC. A group is skipped and reported, never deleted, when

- it holds two different PMCIDs (two Europe PMC records, not one paper twice);
- any copy is curated or registry-confirmed rather than LLM-classified;
- no single copy matches the combined identifiers;
- the copy that would go carries an enrichment, a DOME Registry entry, an abstract or data links the
  keeper lacks -- those are worth a person's judgement, not an automatic delete.

A licence or a higher citation count on the copy that goes is merged into the keeper first, through
`moros_write`'s own `licences` and `citations` modes, so the merge obeys the same allowlists and
leaves its own rollback file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mongo_landscape_export" / "scripts"))
from pid import mint_landscape_pid  # noqa: E402 -- the one minting rule, imported not copied

import moros_write as mw  # noqa: E402
from moros_client import ID_FIELD, Moros  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
KEY_FIELDS = ("pmid", "doi")
DELETE_BATCH = 500

PROJECTION = {
    ID_FIELD: 1,
    "identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1, "identifiers.dome_registry": 1,
    "llm_enrichment.provider": 1, "llm_classification.classification": 1,
    "publication_metadata.citation_count": 1, "publication_metadata.citation_count_updated": 1,
    "publication_metadata.citation_source": 1, "publication_metadata.abstract": 1,
    "source.decision_provenance": 1, "source.access.license": 1, "source.access.open_access": 1,
    "data_links.has_data": 1,
}

SKIP_KINDS = ("different_pmcids", "curated_or_registry", "no_keeper", "needs_review")


# ----------------------------------------------------------------------------- reading the corpus

def _ids(doc: dict) -> dict[str, str | None]:
    ident = doc.get("identifiers") or {}
    return {f: (ident.get(f) or None) for f in ("pmcid", "doi", "pmid")}


def _dig(doc: dict, path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def duplicate_groups(moros: Moros, max_time_ms: int = 1_700_000) -> dict[tuple[str, ...], list[dict]]:
    """Every set of documents sharing a pmid or a doi, keyed by their ids. A group found under both
    keys is the same group and is returned once."""
    groups: dict[tuple[str, ...], list[dict]] = {}
    for field in KEY_FIELDS:
        pipeline = [
            {"$match": {f"identifiers.{field}": {"$nin": [None, ""]}}},
            {"$group": {"_id": {"$toLower": f"$identifiers.{field}"}, "n": {"$sum": 1},
                        "docs": {"$push": "$$ROOT"}}},
            {"$match": {"n": {"$gt": 1}}},
        ]
        for group in moros.collection.aggregate(
                [{"$project": PROJECTION}, *pipeline], allowDiskUse=True, maxTimeMS=max_time_ms):
            docs = group["docs"]
            groups[tuple(sorted(d[ID_FIELD] for d in docs))] = docs
    return groups


# ----------------------------------------------------------------------------- deciding

def combined_identifiers(docs: list[dict]) -> dict[str, str | None]:
    """The identifiers of the paper, taken across its copies: what a fetch that saw all of them
    would carry, and so what its `_id` should be minted from."""
    out: dict[str, str | None] = {}
    for field in ("pmcid", "doi", "pmid"):
        out[field] = next((_ids(d)[field] for d in docs if _ids(d)[field]), None)
    return out


def merge_plan(keeper: dict, losers: list[dict]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """({mode: {leaf path: value}}, blockers). A blocker is something on a copy due for removal that
    no write mode here can move, so the group is left for a person."""
    merges: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []

    if any(_dig(d, "llm_enrichment.provider") for d in losers) and not _dig(keeper, "llm_enrichment.provider"):
        blockers.append("an enrichment")
    if any(_dig(d, "identifiers.dome_registry") for d in losers) and not _dig(keeper, "identifiers.dome_registry"):
        blockers.append("a DOME Registry entry")
    if any(_dig(d, "publication_metadata.abstract") for d in losers) and not _dig(keeper, "publication_metadata.abstract"):
        blockers.append("an abstract")  # publication_metadata.abstract is in no write mode's allowlist
    if any(_dig(d, "data_links.has_data") for d in losers) and not _dig(keeper, "data_links.has_data"):
        blockers.append("data links")

    # "" is "looked up, none disclosed" and None is "never looked up"; either way a copy that carries
    # a real licence is the better answer, because Europe PMC discloses the licence on the PMC record
    # and not on the MEDLINE one, and these pairs are exactly one of each.
    if not _dig(keeper, "source.access.license"):
        donor = next((d for d in losers if _dig(d, "source.access.license")), None)
        if donor is not None:
            merges["licences"] = {
                "source.access.license": _dig(donor, "source.access.license"),
                "source.access.open_access": _dig(donor, "source.access.open_access"),
            }

    keeper_count = _dig(keeper, "publication_metadata.citation_count") or 0
    donor = max(losers, key=lambda d: _dig(d, "publication_metadata.citation_count") or 0, default=None)
    if donor is not None and (_dig(donor, "publication_metadata.citation_count") or 0) > keeper_count:
        merges["citations"] = {
            "publication_metadata.citation_count": _dig(donor, "publication_metadata.citation_count"),
            "publication_metadata.citation_count_updated": _dig(donor, "publication_metadata.citation_count_updated"),
            "publication_metadata.citation_source": _dig(donor, "publication_metadata.citation_source"),
        }
    return merges, blockers


def classify_group(docs: list[dict]) -> dict[str, Any]:
    """What to do with one group: `kind`, the keeper, the copies to remove, and any merge first."""
    pmcids = {(_ids(d)["pmcid"] or "").upper() for d in docs if _ids(d)["pmcid"]}
    if len(pmcids) > 1:
        return {"kind": "different_pmcids", "keeper": None, "losers": [], "merges": {},
                "why": f"{len(pmcids)} distinct PMCIDs: separate Europe PMC records, not one paper twice"}
    if any((_dig(d, "source.decision_provenance") or "llm") != "llm" for d in docs):
        return {"kind": "curated_or_registry", "keeper": None, "losers": [], "merges": {},
                "why": "a copy is curated or registry-confirmed"}

    want = mint_landscape_pid(**combined_identifiers(docs))
    matches = [d for d in docs if d[ID_FIELD] == want]
    if len(matches) != 1:
        return {"kind": "no_keeper", "keeper": None, "losers": [], "merges": {},
                "why": f"{len(matches)} copies match the combined identifiers ({want})"}

    keeper, losers = matches[0], [d for d in docs if d[ID_FIELD] != want]
    merges, blockers = merge_plan(keeper, losers)
    if blockers:
        return {"kind": "needs_review", "keeper": keeper[ID_FIELD], "losers": [], "merges": {},
                "why": "the copy that would go carries " + ", ".join(blockers) + " the keeper lacks"}
    return {"kind": "merge_then_remove" if merges else "plain",
            "keeper": keeper[ID_FIELD], "losers": [d[ID_FIELD] for d in losers], "merges": merges,
            "why": "merge " + ", ".join(sorted(merges)) + " first" if merges else "nothing unique on the copy"}


def plan_runs(groups: dict[tuple[str, ...], list[dict]], limit: int | None = None) -> tuple[list[dict], dict[str, int]]:
    """Classified groups in a stable order, and a count per kind. `limit` caps how many groups are
    acted on, for a trial; skipped kinds never count against it."""
    decided, counts, acted = [], {}, 0
    for key in sorted(groups):
        decision = classify_group(groups[key])
        counts[decision["kind"]] = counts.get(decision["kind"], 0) + 1
        removable = decision["kind"] in ("plain", "merge_then_remove")
        if removable and limit is not None and acted >= limit:
            decision = {**decision, "kind": "held_back_by_limit", "losers": [], "merges": {}}
            counts["held_back_by_limit"] = counts.get("held_back_by_limit", 0) + 1
            counts[classify_group(groups[key])["kind"]] -= 1
        elif removable:
            acted += 1
        decided.append({**decision, "group": list(key)})
    return decided, counts


# ----------------------------------------------------------------------------- writing

def apply_merges(moros: Moros, decided: list[dict], run_id: str, dry_run: bool) -> dict[str, dict]:
    """Each mode's updates through SafeWriter, which validates them against that mode's allowlist
    and writes its own rollback file."""
    by_mode: dict[str, list[tuple[str, dict]]] = {}
    for decision in decided:
        for mode, fields in decision.get("merges", {}).items():
            by_mode.setdefault(mode, []).append((decision["keeper"], fields))
    results = {}
    for mode, updates in sorted(by_mode.items()):
        writer = mw.SafeWriter(moros, mode, run_id=f"{run_id}_{mode}", dry_run=dry_run)
        results[mode] = writer.apply(updates).as_dict()
    return results


def snapshot_and_delete(moros: Moros, ids: list[str], run_id: str) -> tuple[Path, int]:
    """Writes every document about to go, whole, then deletes exactly those ids. The snapshot is the
    only way back, so it is written and flushed before the first delete."""
    path = mw.ROLLBACK_DIR / f"{run_id}.deleted.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for batch in _batched(ids, DELETE_BATCH):
            for doc in moros.collection.find({ID_FIELD: {"$in": batch}}):
                f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
                written += 1
        f.flush()
    if written != len(ids):
        raise SystemExit(f"snapshot holds {written:,} of {len(ids):,} documents -- nothing deleted")

    deleted = 0
    for batch in _batched(ids, DELETE_BATCH):
        deleted += moros.collection.delete_many({ID_FIELD: {"$in": batch}}).deleted_count
    return path, deleted


def restore(moros: Moros, path: Path, confirm: bool) -> None:
    """Puts back what a run deleted, from its own snapshot."""
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    present = moros.existing_ids([d[ID_FIELD] for d in docs])
    missing = [d for d in docs if d[ID_FIELD] not in present]
    print(f"restore {path.name}: {len(docs):,} snapshotted, {len(present):,} already present, "
          f"{len(missing):,} to insert")
    if not confirm:
        print("  DRY RUN -- pass --confirm to insert them.")
        return
    for batch in _batched(missing, DELETE_BATCH):
        moros.collection.insert_many(batch, ordered=False)
    print(f"  inserted {len(missing):,}; corpus now {moros.count():,}")


def _batched(items: Iterable, size: int) -> Iterable[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="Actually merge and delete. Off by default.")
    parser.add_argument("--limit", type=int, default=None, help="Act on only the first N removable groups.")
    parser.add_argument("--restore", type=Path, default=None, help="A <run_id>.deleted.jsonl to put back.")
    parser.add_argument("--report", type=Path, default=None, help="Where to write the classification.")
    args = parser.parse_args()

    run_id = mw.new_run_id("resolve_duplicates")
    with Moros.from_env() as moros:
        print(f"[{run_id}] target {moros.describe()}")
        if args.restore:
            restore(moros, args.restore, args.confirm)
            return 0

        groups = duplicate_groups(moros)
        decided, counts = plan_runs(groups, args.limit)
        ids = [i for d in decided for i in d["losers"]]
        print(f"[{run_id}] {len(groups):,} duplicate group(s)")
        for kind, n in sorted(counts.items()):
            print(f"    {n:>7,}  {kind}")
        print(f"[{run_id}] {len(ids):,} document(s) to remove, "
              f"{sum(1 for d in decided if d['merges']):,} group(s) to merge first")

        report = {"run_id": run_id, "target": moros.describe(), "dry_run": not args.confirm,
                  "groups": len(groups), "counts": counts, "documents_to_remove": len(ids),
                  "skipped": [d for d in decided if d["kind"] in SKIP_KINDS], "corpus_before": moros.count()}

        if not args.confirm:
            apply_merges(moros, decided, run_id, dry_run=True)
            print(f"[{run_id}] DRY RUN -- nothing written. Re-run with --confirm (--limit N first).")
        else:
            report["merges"] = apply_merges(moros, decided, run_id, dry_run=False)
            snapshot, deleted = snapshot_and_delete(moros, ids, run_id)
            report.update({"snapshot": str(snapshot), "deleted": deleted, "corpus_after": moros.count(),
                           "restore_command": f"python3 resolve_duplicates.py --restore {snapshot} --confirm"})
            print(f"[{run_id}] deleted {deleted:,}; corpus {report['corpus_before']:,} -> "
                  f"{report['corpus_after']:,}")
            print(f"[{run_id}] snapshot -> {snapshot}")

    path = args.report or OUTPUT_DIR / f"{run_id}.report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[{run_id}] report -> {path}")
    print(f"[{run_id}] next: python3 verify_corpus.py --expect-count <corpus after>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
