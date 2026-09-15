"""Upserts whole new documents into moros via `mongoimport`, with the guards around it.

The counterpart to `load_fields.py`: this one is for JSONL files of *complete* documents (the
curated merge, and each future incremental batch). `mongoimport --mode upsert --upsertFields _id`
is what both repos' roadmaps specify, and it is the right tool -- it streams, it handles a 3GB
file, and the deterministic UUID5 `_id` from `pid.py` is what makes upsert idempotent instead of
duplicating.

**Never drop-and-reimport.** Dropping `Content` destroys `positives_text`, which costs ~3 minutes
of tokenising plus a 1.5GB collection read to rebuild -- and while it is gone `observatory-ws`
silently degrades to a regex scan with no error logged anywhere. There is no code path here that
drops anything.

What this adds around mongoimport:

- **A reconciliation gate.** For the curated merge it reads `curated_conversion_report.json` and
  refuses to run if the builder's own count reconciliation did not pass. A load is not the place
  to discover that the input's premise has drifted.
- **A pre-flight split of inserts from updates.** Every `_id` in the file is checked against the
  collection first, so the run says up front how many documents are genuinely new and how many
  already exist. For the curated merge the expected answer is "all new, none existing" -- these
  records have never been in the corpus -- and anything else is worth stopping for.
- **A rollback spec.** The inverse of inserting is deleting, so the spec lists exactly the `_id`s
  that did not exist beforehand. `--reverse` deletes only ids listed in a spec this tool wrote.
  Documents that already existed are reported and *not* made reversible, because restoring their
  prior content is `load_fields.py`'s snapshot mechanism, not this one's.
- **The `record_modified` stamp (v1.6.0).** Every document written gets this run's stamp, so a
  harvester asking for records changed since its last visit sees it. The stamp is ignored when a
  resumed load compares a document with what moros already holds: only the stamp would differ.

    python3 load_documents.py --input ../../mongo_landscape_export/output/curated_records.jsonl
    python3 load_documents.py --input ... --limit 100 --confirm
    python3 load_documents.py --input ... --confirm
    python3 load_documents.py --reverse ../output/rollback/<run_id>.inserted.json --confirm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from pymongo import ReplaceOne

from link_identifiers import malformed_links
from moros_client import ID_FIELD, Moros, load_env
from moros_write import (
    RECORD_MODIFIED_PATH,
    REPORT_DIR,
    ROLLBACK_DIR,
    new_run_id,
    record_modified_stamp,
    sha256_file,
    utc_now_iso,
)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
REPO_DIR = FOLDER_DIR.parent

DEFAULT_INPUT = REPO_DIR / "mongo_landscape_export" / "output" / "curated_records.jsonl"
DEFAULT_GATE_REPORT = REPO_DIR / "mongo_landscape_export" / "output" / "curated_conversion_report.json"

# Upsert is idempotent, so retrying a partial import only re-sets documents to the same values.
MAX_IMPORT_ATTEMPTS = 3
COMPARE_FETCH_BATCH = 500
UPSERT_BATCH_SIZE = 500


def read_ids(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.append(json.loads(line)[ID_FIELD])
            if limit is not None and len(ids) >= limit:
                break
    return ids


def check_gate(gate_report: Path) -> dict:
    """The builder records whether its own count reconciliation passed. Trusting a JSONL file
    without checking that is how a silently-shrunk input gets loaded."""
    if not gate_report.exists():
        print(f"  no reconciliation report at {gate_report} -- skipping the gate")
        return {}
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    rec = report.get("reconciliation", {})
    if rec and not rec.get("passed"):
        raise SystemExit(
            f"refusing to load: {gate_report.name} records a FAILED reconciliation "
            f"({rec}). Fix the input before loading it."
        )
    if rec:
        print(f"  reconciliation gate: passed "
              f"({rec.get('excluded_bulk_rows'):,} excluded bulk rows, "
              f"{rec.get('matched_canonical_rows'):,} matched canonical rows)")
    if report.get("conflicted_papers_held_for_review"):
        print(f"  note: {report['conflicted_papers_held_for_review']} paper(s) were held back for "
              f"curation review and are deliberately NOT in this file")
    return report


def slice_input(path: Path, limit: int) -> Path:
    """A real N-document trial needs a real N-document file, since mongoimport has no --limit."""
    tmp = Path(tempfile.mkdtemp(prefix="moros_load_")) / f"{path.stem}.first{limit}.jsonl"
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        written = 0
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= limit:
                break
    return tmp


def run_mongoimport(env: dict[str, str], file_path: Path) -> tuple[str, int]:
    cmd = [
        "mongoimport",
        "--uri", env["MONGODB_URI"],
        "--db", env["MONGODB_DB"],
        "--collection", env["MONGODB_COLLECTION"],
        "--file", str(file_path),
        "--mode", "upsert",
        "--upsertFields", ID_FIELD,
    ]
    # The URI is the only sensitive part; show the command with it redacted so the log is useful
    # without leaking the internal host.
    shown = list(cmd)
    shown[shown.index(env["MONGODB_URI"])] = "<MONGODB_URI>"
    print(f"  $ {' '.join(shown)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print("  " + output.strip().replace("\n", "\n  "))
    # Deliberately does NOT raise on a non-zero exit. Observed for real against this server
    # (2026-09-03): mongoimport wrote all 100 documents, then lost its connection during
    # teardown and reported both `Failed: ... use of closed network connection` and
    # "0 document(s) imported successfully" -- while the collection count had already gone up by
    # exactly 100. Trusting the exit code would have declared a successful load a failure, and
    # trusting the summary line would have under-reported it. The collection count is the only
    # honest authority, so the caller checks that instead. Upsert is idempotent, so a retry after
    # a genuinely partial import is always safe.
    return output, result.returncode


def iter_documents(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            if limit is not None and n >= limit:
                return
            yield json.loads(line)


def find_malformed_data_links(path: Path, limit: int | None) -> list[tuple[str, tuple]]:
    """(document _id, malformed link) for every link in the input that must not reach moros.

    `build_data_links.py` chooses identifiers and `build_staged_documents.py` carries them into the
    batch's documents unchanged, so this should always be empty; a non-empty answer means the JSONL
    was built from a stale or hand-edited data-links file. `link_identifiers.malformed_links` is the
    one definition, shared with `load_fields.py` and the build itself."""
    found: list[tuple[str, tuple]] = []
    for doc in iter_documents(path, limit):
        for link in malformed_links(doc.get("data_links")):
            found.append((doc.get(ID_FIELD), link))
    return found


def _without_stamp(doc: dict | None) -> dict | None:
    """The document minus `record_modified`, for comparing content across two load runs."""
    if doc is None:
        return None
    return {k: v for k, v in doc.items() if k != RECORD_MODIFIED_PATH}


def upsert_via_pymongo(
    moros: Moros, path: Path, limit: int | None, existing: set[str],
    allow_replace_existing: bool, stamp: str,
) -> dict:
    """Batched `ReplaceOne(upsert=True)` -- the same semantics as
    `mongoimport --mode upsert --upsertFields _id`, over the driver that actually holds a
    connection to this server.

    Why not mongoimport, which both roadmaps name: measured against this server on 2026-09-03, it
    stalls at ~17% of a 21.4MB file, reports `use of closed network connection`, and dies having
    written exactly one 1,000-document batch -- three attempts, same result. The same host took
    811,036 pymongo bulk writes in the citation load with zero batch errors. The requirement that
    actually matters is "upsert, never drop-and-reimport", and this satisfies it exactly.

    **Replacing an existing document is refused by default.** A whole-document replace blanks
    whatever the new document does not mention -- precisely the hazard the sibling roadmap warns
    about ("a classification refresh must not blank an `llm_enrichment` group a later enrichment
    run has populated"). But an interrupted load leaves documents that are byte-identical to what
    would be rewritten, so those are compared and skipped rather than blocked: resuming is safe by
    construction, while genuinely clobbering a differing document still needs saying so.
    """
    stats = {"inserted": 0, "identical_skipped": 0, "replaced": 0, "differing": []}
    batch: list[ReplaceOne] = []
    documents = list(iter_documents(path, limit))
    progress = tqdm(total=len(documents), desc="upsert", unit="doc")

    # Pre-fetch the current state of every document that already exists, in batches. Fetching them
    # one at a time is one VPN round-trip per document: 6,179 of them took longer than a 9-minute
    # timeout, while the comparison itself is instant. This is the same batched-$in shape
    # `Moros.existing_ids` uses, and it turns thousands of round-trips into a handful.
    current_by_id: dict[str, dict] = {}
    to_fetch = [d[ID_FIELD] for d in documents if d[ID_FIELD] in existing]
    if to_fetch:
        for start in tqdm(range(0, len(to_fetch), COMPARE_FETCH_BATCH), desc="compare-fetch",
                          unit="batch", leave=False):
            chunk = to_fetch[start:start + COMPARE_FETCH_BATCH]
            for current in moros.collection.find({ID_FIELD: {"$in": chunk}}):
                current_by_id[current[ID_FIELD]] = current

    def flush() -> None:
        if not batch:
            return
        moros.collection.bulk_write(batch, ordered=False)
        batch.clear()

    for doc in documents:
        doc_id = doc[ID_FIELD]
        if doc_id in existing:
            if _without_stamp(current_by_id.get(doc_id)) == _without_stamp(doc):
                stats["identical_skipped"] += 1
                progress.update(1)
                continue
            if not allow_replace_existing:
                stats["differing"].append(doc_id)
                progress.update(1)
                continue
            stats["replaced"] += 1
        else:
            stats["inserted"] += 1
        batch.append(ReplaceOne({ID_FIELD: doc_id}, {**doc, RECORD_MODIFIED_PATH: stamp}, upsert=True))
        if len(batch) >= UPSERT_BATCH_SIZE:
            flush()
        progress.update(1)
    flush()
    progress.close()
    return stats


def run(input_path: Path, gate_report: Path, confirm: bool, limit: int | None,
        allow_replace_existing: bool = False) -> None:
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist")

    run_id = new_run_id("load_documents")
    stamp = record_modified_stamp()
    env = load_env()
    print(f"[{run_id}] input {input_path.name} ({input_path.stat().st_size / 1e6:.1f} MB)")
    report = check_gate(gate_report)

    ids = read_ids(input_path, limit)
    print(f"[{run_id}] {len(ids):,} documents in scope"
          + (f" (--limit {limit})" if limit else ""))
    if len(set(ids)) != len(ids):
        raise SystemExit("the input contains duplicate _ids -- refusing to load")

    malformed = find_malformed_data_links(input_path, limit)
    if malformed:
        raise SystemExit(
            f"[{run_id}] refusing to load, nothing written: {len(malformed):,} malformed data link(s) "
            f"in {len({doc_id for doc_id, _ in malformed}):,} document(s), first {malformed[:3]} -- "
            f"rebuild the data links with build_data_links.py, then the documents"
        )

    with Moros.from_env() as moros:
        print(f"[{run_id}] target {moros.describe()}")
        before = moros.count()
        existing = moros.existing_ids(tqdm(ids, desc="pre-flight", unit="id"))
        new_ids = [i for i in ids if i not in existing]
        print(f"[{run_id}] {len(new_ids):,} genuinely new, {len(existing):,} already present")
        if existing:
            print(f"[{run_id}] NOTE the {len(existing):,} existing documents will be UPDATED in "
                  f"place. Their prior content is not captured by this tool's rollback -- only "
                  f"inserts are reversible here.")

        print(f"[{run_id}] indexes present: {sorted(moros.indexes())}")

        if not confirm:
            print(f"\n[{run_id}] DRY RUN -- nothing written. First 3 documents:")
            with input_path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    d = json.loads(line)
                    print(f"    {d[ID_FIELD]}  {d['source']['decision_provenance']:<19} "
                          f"{d['llm_classification']['classification']:<15} "
                          f"{(d['publication_metadata']['title'] or '')[:60]}")
            print(f"\n[{run_id}] would upsert {len(ids):,} documents; corpus "
                  f"{before:,} -> {before + len(new_ids):,}")
            print(f"[{run_id}] re-run with --confirm to write.")
            return

        ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        spec_path = ROLLBACK_DIR / f"{run_id}.inserted.json"
        spec_path.write_text(json.dumps({
            "run_id": run_id,
            "kind": "inserted-ids",
            "why": "the inverse of an insert is a delete; only ids absent beforehand are listed",
            "target": moros.describe(),
            "taken_at": utc_now_iso(),
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "inserted_ids": new_ids,
            "updated_ids": sorted(existing),
            "reverse_command": f"python3 load_documents.py --reverse {spec_path} --confirm",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[{run_id}] rollback spec ({len(new_ids):,} reversible inserts) -> {spec_path}")

        print(f"[{run_id}] {RECORD_MODIFIED_PATH} -> {stamp} on every document written")
        stats = upsert_via_pymongo(
            moros, input_path, limit, existing, allow_replace_existing, stamp
        )
        after = moros.count()
        delta = after - before
        still_absent = len(new_ids) - len(moros.existing_ids(new_ids))

        print(f"\n[{run_id}] inserted {stats['inserted']:,}, "
              f"replaced {stats['replaced']:,}, "
              f"skipped {stats['identical_skipped']:,} already byte-identical")
        if stats["differing"]:
            print(f"[{run_id}] !! {len(stats['differing']):,} document(s) already exist with "
                  f"DIFFERENT content and were left untouched. Replacing them would blank any "
                  f"field the new document does not carry. Re-run with "
                  f"--allow-replace-existing if that is genuinely intended. "
                  f"First few: {stats['differing'][:3]}")
        print(f"[{run_id}] corpus {before:,} -> {after:,} (delta {delta:+,}); "
              f"{still_absent:,} of the {len(new_ids):,} new documents still absent")
        if still_absent:
            raise SystemExit(
                f"[{run_id}] {still_absent:,} documents did not land. Nothing was rolled back; "
                f"re-running is safe and idempotent."
            )

        if delta != len(new_ids):
            print(f"[{run_id}] !! expected a delta of {len(new_ids):+,}, got {delta:+,} -- "
                  f"investigate before treating this load as complete")

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{run_id}.report.json"
        report_path.write_text(json.dumps({
            "run_id": run_id,
            "target": moros.describe(),
            "finished_at": utc_now_iso(),
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "documents_in_scope": len(ids),
            "inserted": len(new_ids),
            "updated": len(existing),
            "count_before": before,
            "count_after": after,
            "delta": delta,
            "delta_matches_expected": delta == len(new_ids),
            "writer": "pymongo ReplaceOne(upsert=True)",
            "record_modified": stamp,
            "upsert_stats": {k: (len(v) if isinstance(v, list) else v)
                             for k, v in stats.items()},
            "rollback": str(spec_path),
            "source_report": report or None,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[{run_id}] report -> {report_path}")
        print(f"\n[{run_id}] next: python3 ensure_indexes.py && python3 verify_corpus.py")


def reverse(spec_path: Path, confirm: bool) -> None:
    """Deletes only the ids this tool recorded as inserts. The single delete path in this folder,
    deliberately unable to name anything the spec does not."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("kind") != "inserted-ids":
        raise SystemExit(f"{spec_path} is not an inserted-ids rollback spec")
    ids = spec["inserted_ids"]
    print(f"reverse {spec_path.name}: {len(ids):,} inserted ids from run {spec['run_id']}")
    if spec.get("updated_ids"):
        print(f"  NOTE {len(spec['updated_ids']):,} documents were UPDATED by that run and are "
              f"not reversed here -- their prior content was never captured.")
    with Moros.from_env() as moros:
        print(f"  target {moros.describe()}")
        present = moros.existing_ids(ids)
        print(f"  {len(present):,} of them are currently present")
        if not confirm:
            print("  DRY RUN -- pass --confirm to delete them.")
            return
        result = moros.collection.delete_many({ID_FIELD: {"$in": list(present)}})
        print(f"  deleted {result.deleted_count:,}; corpus now {moros.count():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Load only the first N documents -- a real, reversible trial.")
    parser.add_argument("--allow-replace-existing", action="store_true",
                        help="Replace documents that already exist with DIFFERENT content. "
                             "A whole-document replace blanks anything the new document omits.")
    parser.add_argument("--reverse", type=Path, default=None,
                        help="An inserted-ids rollback spec to undo.")
    args = parser.parse_args()
    if args.reverse:
        reverse(args.reverse, args.confirm)
    else:
        run(args.input, args.gate_report, args.confirm, args.limit,
            args.allow_replace_existing)


if __name__ == "__main__":
    main()
