"""Step 24 phase 2: converts the human-curated and registry-confirmed records into MongoDB
documents, so the corpus finally contains this project's own ground truth.

Why this exists: Step 23a deliberately excluded every already-curated record from LLM
classification -- they had human-final decisions, so paying a model to re-decide them added no
signal. That was correct. What never happened is the merge back in, so the 6,197 highest-confidence
records in the project -- including 378 confirmed DOME Registry entries and the AlphaFold 2 paper
itself -- are the *only* records absent from the published corpus. See
`../../FINALISATION_ROADMAP.md` for the full diagnosis.

The order of operations below is load-bearing, and one step is a genuine trap:

1. **Recompute the exclusion, never trust a stored list.** Streams `bulk_candidates.csv` against
   `canonical_dataset.csv` using the same rule `llm_classify/sampling.py::
   select_bulk_pool_excluding_curated` applied (a bulk row is curated if *any* of its pmcid / pmid
   / doi appears among the canonical set's identifiers). Must reproduce 6,197 excluded bulk rows
   and 6,356 matched canonical rows, and hard-fails if it does not -- if that number has moved,
   something upstream changed and this script's premise needs re-checking before it writes
   anything.

2. **Backfill identifiers and metadata from the bulk pool BEFORE minting pids.** Two reasons, and
   the first is the trap:

   - `pid.py` mints its UUID5 from `pmcid > doi > pmid`, so a row carrying only its own matched key
     would mint a *different* `_id` than the same paper reached by a different key. AlphaFold 2 is
     literally two canonical rows, one keyed `PMCID:PMC8371605` and one keyed
     `DOI:10.1038/s41586-021-03819-2`. Measured on the real file, all 8,614 canonical rows already
     carry the full triple and 8,439 papers mint exactly 8,439 pids -- so today this is safe -- but
     the invariant is asserted rather than assumed, because a future dataset need not be so tidy.
     Taking identifiers from the bulk pool additionally makes these pids *identical* to the ones
     the bulk path would have minted, which is what makes the upsert idempotent.
   - Curated rows carry empty `mesh_headings`, `pub_types` and `keywords_author` (confirmed on both
     AlphaFold rows) because they came from PDF and registry sources, not an EPMC
     `resultType=core` fetch. The bulk pool has that metadata for these same papers. Without the
     join, ~6k records land with empty facets and silently distort every facet count in the UI.

3. **Dedupe on the minted pid**, not on `record_id`: `record_id` is per *canonical row*, and the
   whole point is that one paper can have several. Conflicting labels under one pid hard-fail.

4. **Drop the 7 `skipped` records.** `schema.py`'s `classification` enum has no such value, and a
   curator declining to judge is not a verdict. The 8 `undeterminable` are kept -- that *is* a
   valid value, and `source.decision_provenance` keeps a human "can't tell" distinguishable from a
   model's.

Read-only against every input. Writes only into `output/`.

    python3 build_curated_documents.py --report-only    # reproduce the counts, write nothing
    python3 build_curated_documents.py                  # -> output/curated_records.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from citations_index import load_citation_index, lookup_citation
from pid import mint_landscape_pid
from schema import PROVENANCE_HUMAN, PROVENANCE_REGISTRY, build_curated_document

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
REPO_DIR = FOLDER_DIR.parent

DEFAULT_CANONICAL = REPO_DIR / "data" / "processed" / "canonical_dataset.csv"
DEFAULT_BULK = REPO_DIR / "data" / "interim" / "bulk_candidates.csv"
DEFAULT_LICENSING = REPO_DIR / "epmc_licensing" / "output" / "epmc_pmid_licensing.csv"
DEFAULT_CITATIONS = REPO_DIR / "moros_pipeline" / "output" / "epmc_citations.csv"
DEFAULT_OUT_JSONL = FOLDER_DIR / "output" / "curated_records.jsonl"
DEFAULT_OUT_CSV = FOLDER_DIR / "output" / "curated_normalised.csv"
DEFAULT_OUT_REPORT = FOLDER_DIR / "output" / "curated_conversion_report.json"
DEFAULT_EXCLUDED_CSV = REPO_DIR / "data" / "processed" / "landscape_excluded_curated.csv"
DEFAULT_CONFLICTS_CSV = REPO_DIR / "data" / "processed" / "curated_merge_conflicts.csv"

# The event logs that carry a real, dated human decision. Latest wins.
CURATION_EVENT_LOGS = (
    REPO_DIR / "data" / "processed" / "curation_events.csv",
    REPO_DIR / "data" / "processed" / "original_cohort_review_events.csv",
    REPO_DIR / "data" / "processed" / "cross_curate_resolution_events.csv",
    REPO_DIR / "data" / "processed" / "llm_seed_review_events.csv",
)

ID_COLUMNS = ("pmcid", "pmid", "doi")
# Copied from the bulk pool onto the curated row. The identifier triple comes first because
# minting depends on it (see the module docstring).
BULK_BACKFILL_COLUMNS = (
    "pmcid", "pmid", "doi",
    "title", "abstract", "authors", "year", "journal",
    "mesh_headings", "pub_types", "keywords_author",
    "is_open_access", "fulltext_available", "abstract_source", "metadata_repair_sources",
)

# What the normalised intermediate carries -- the landscape file's names, so `schema.py`'s shared
# group builders work on it unchanged, plus the five curation-specific columns.
NORMALISED_COLUMNS = [
    "pid", "record_id", "canonical_key",
    "pmid", "pmcid", "doi",
    "title", "abstract", "authors", "year", "journal",
    "citation_count", "citation_count_updated", "citation_source",
    "mesh_headings", "pub_types", "keywords_author",
    "is_open_access", "fulltext_available", "abstract_source", "metadata_repair_sources",
    "license", "license_checked", "epmc_is_open_access",
    "label", "label_confidence",
    "curation_rationale", "curation_timestamp", "curation_batch_id",
]

# Measured against the real files, 2026-09-03. A mismatch stops the run.
EXPECTED_EXCLUDED_BULK_ROWS = 6197
EXPECTED_MATCHED_CANONICAL_ROWS = 6356

CITATION_SOURCE = "europepmc"

PUBLISHABLE_CONFIDENCE = (PROVENANCE_HUMAN, PROVENANCE_REGISTRY)
EXCLUDED_LABELS = ("skipped",)

# registry_confirmed wins a tie: it is an externally verifiable DOME Registry entry.
_CONFIDENCE_RANK = {PROVENANCE_REGISTRY: 0, PROVENANCE_HUMAN: 1}


def _blank(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_existing_ids(canonical_path: Path) -> set[str]:
    """Every non-blank pmcid / pmid / doi in the curated set -- the identical set
    `_load_existing_ids(canonical_path)` builds everywhere else in this codebase, so "already
    curated" means the same thing here as it did when the exclusion ran."""
    ids: set[str] = set()
    with canonical_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            for col in ID_COLUMNS:
                value = _blank(row.get(col))
                if value:
                    ids.add(value)
    return ids


def scan_bulk_pool(
    bulk_path: Path, existing_ids: set[str]
) -> tuple[dict[str, dict], int, list[dict]]:
    """One streaming pass over the 1.83GB bulk pool. Returns:

    - `by_id`: every identifier of every excluded bulk row -> that row's backfill metadata, so a
      canonical row can find its bulk twin by whichever key it happens to share.
    - the count of excluded bulk rows (must be EXPECTED_EXCLUDED_BULK_ROWS).
    - the excluded rows themselves, for the audit artifact (F5) that makes this exclusion visible
      on disk instead of only in a log line -- the absence of which is why the gap went unnoticed.
    """
    by_id: dict[str, dict] = {}
    excluded_rows: list[dict] = []
    n_excluded = 0

    with bulk_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in tqdm(csv.DictReader(f), desc="bulk pool", unit="row"):
            row_ids = [_blank(row.get(c)) for c in ID_COLUMNS]
            if not any(i and i in existing_ids for i in row_ids):
                continue
            n_excluded += 1
            payload = {c: _blank(row.get(c)) for c in BULK_BACKFILL_COLUMNS}
            for value in row_ids:
                # First writer wins: ~9,050 bulk rows share an identity (documented in Step
                # 23a(b)), and taking the first keeps this deterministic across re-runs.
                if value and value not in by_id:
                    by_id[value] = payload
            excluded_rows.append({
                "pmcid": row_ids[0], "pmid": row_ids[1], "doi": row_ids[2],
                "title": _blank(row.get("title"))[:300],
                "journal": _blank(row.get("journal")),
                "year": _blank(row.get("year")),
                "source_name": _blank(row.get("source_name")),
            })
    return by_id, n_excluded, excluded_rows


def load_curation_timestamps(paths: tuple[Path, ...]) -> dict[str, str]:
    """record_id -> the latest real curation-event timestamp. Preferred over
    `canonical_dataset.csv`'s `updated_at`, which records when the dataset was last materialised,
    not when a human actually decided."""
    latest: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            print(f"  WARNING {path.name} missing -- skipping")
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                rid = _blank(row.get("record_id"))
                ts = _blank(row.get("timestamp"))
                if rid and ts and ts > latest.get(rid, ""):
                    latest[rid] = ts
    return latest


def load_licensing(path: Path) -> dict[str, tuple[str, str]]:
    """Same shape as `join_license.py::load_licensing` -- pmid -> (license, is_open_access)."""
    table: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[_blank(row.get("pmid"))] = (row.get("license") or "", row.get("is_open_access") or "")
    return table


def load_citations(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """The identifier-keyed index, not a pid-keyed one. These records were deliberately absent
    from the landscape corpus, so they have no row in `pid_citations.csv` (which is built by
    joining against the landscape CSV) -- their counts have to be found by identifier."""
    if not Path(path).exists():
        print(f"  WARNING {path} missing -- curated records will have no citation counts")
        return {}
    return load_citation_index(Path(path))


def _merge_sources(rows: list[dict]) -> str:
    """Union of the `sources` JSON across every canonical row for one paper, de-duplicated on
    (source_name, matched_on). One paper matched under two keys was recorded by two sources; both
    are real provenance and both belong in the rationale."""
    merged: dict[tuple[str, str], dict] = {}
    for row in rows:
        try:
            for source in json.loads(row.get("sources") or "[]"):
                key = (_blank(source.get("source_name")), _blank(source.get("matched_on")))
                merged.setdefault(key, source)
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return json.dumps(list(merged.values()))


def build_rationale(row: dict) -> str:
    """A factual provenance sentence assembled from the record's own columns -- never an invented
    justification. `schema.py` writes this into `llm_classification.rationale`, whose v1.1.0
    description claimed rationales are always model-generated; v1.2.0 corrects that claim and
    `source.decision_provenance` is what tells a reader which kind this is."""
    confidence = _blank(row.get("label_confidence"))
    parts: list[str] = []

    try:
        sources = json.loads(row.get("sources") or "[]")
        names = sorted({_blank(s.get("source_name")) for s in sources if _blank(s.get("source_name"))})
        matched_on = sorted({_blank(s.get("matched_on")) for s in sources if _blank(s.get("matched_on"))})
    except (json.JSONDecodeError, AttributeError, TypeError):
        names, matched_on = [], []

    if confidence == PROVENANCE_REGISTRY:
        parts.append("Confirmed DOME Registry entry")
    else:
        parts.append("Human-curated decision")
    curator = _blank(row.get("original_cohort_review_curator")) or _blank(row.get("cross_curate_curator"))
    if curator:
        parts.append(f"curator {curator}")
    if names:
        parts.append(f"sources: {', '.join(names)}")
    if matched_on:
        parts.append(f"matched on {', '.join(matched_on)}")
    if _blank(row.get("cross_curate_final_label")):
        parts.append("label resolved through cross-curation review")
    for field in ("cross_curate_notes", "notes", "curation_tag"):
        note = _blank(row.get(field))
        if note:
            parts.append(f"{field.replace('_', ' ')}: {note}")
            break
    return "; ".join(parts) + "."


def sort_key(row: dict, timestamps: dict[str, str]) -> tuple:
    """Which canonical row wins when several map to one pid: most recent real curation decision,
    then registry_confirmed over human_curated, then record_id so re-runs are identical."""
    rid = _blank(row.get("record_id"))
    return (
        _blank(timestamps.get(rid)) or _blank(row.get("updated_at")),
        -_CONFIDENCE_RANK.get(_blank(row.get("label_confidence")), 9),
        rid,
    )


def _resolution_key(row: dict, timestamps: dict[str, str]) -> tuple:
    """Which row's verdict survives a label conflict. Human judgement first, then recency.

    Deliberately the inverse of `_CONFIDENCE_RANK`'s use in the *agreeing* case, and both are
    right: when rows agree, the strongest provenance claim is worth keeping (a registry entry that
    a human also confirmed is still a registry entry); when rows disagree, the human who read the
    paper and recorded a contrary reason is the one being trusted."""
    rid = _blank(row.get("record_id"))
    human_first = 0 if _blank(row.get("label_confidence")) == PROVENANCE_HUMAN else 1
    return (
        -human_first,  # human_curated sorts above registry_confirmed under max()
        _blank(timestamps.get(rid)) or _blank(row.get("updated_at")),
        rid,
    )


def _resolution_reason(winner: dict, rows: list[dict]) -> str:
    confidences = {_blank(r.get("label_confidence")) for r in rows}
    if confidences == {PROVENANCE_HUMAN, PROVENANCE_REGISTRY}:
        if _blank(winner.get("label_confidence")) == PROVENANCE_HUMAN:
            return "human review overrode the registry entry; published as human_curated"
        return "registry entry retained"
    return "most recent human curation decision"


def normalise(
    canonical_rows: list[dict],
    by_id: dict[str, dict],
    timestamps: dict[str, str],
    licensing: dict[str, tuple[str, str]],
    citations: dict[str, tuple[str, str, str]],
    batch_id: str,
) -> tuple[list[dict], dict, list[dict]]:
    """canonical rows -> normalised rows ready for `schema.build_curated_document`, one per pid.

    Returns `(normalised, stats, conflicts)`. Papers whose canonical rows disagree on the label are
    held back and returned in `conflicts` for human review rather than resolved here."""
    stats = {
        "canonical_matched": len(canonical_rows),
        "dropped_label": 0,
        "dropped_confidence": 0,
        "no_bulk_twin": 0,
        "backfilled_facets": 0,
        "deduped_away": 0,
        "licence_matched": 0,
        "citation_matched": 0,
        "conflicts_resolved": 0,
        "provenance_upgraded": 0,
    }
    candidates: dict[str, list[dict]] = {}

    for row in canonical_rows:
        label = _blank(row.get("label"))
        confidence = _blank(row.get("label_confidence"))
        if label in EXCLUDED_LABELS:
            stats["dropped_label"] += 1
            continue
        if confidence not in PUBLISHABLE_CONFIDENCE:
            stats["dropped_confidence"] += 1
            continue

        merged = dict(row)
        twin = None
        for col in ID_COLUMNS:
            value = _blank(row.get(col))
            if value and value in by_id:
                twin = by_id[value]
                break
        if twin is None:
            stats["no_bulk_twin"] += 1
        else:
            had_facets = any(_blank(row.get(c)) not in ("", "[]") for c in
                             ("mesh_headings", "pub_types", "keywords_author"))
            for col, value in twin.items():
                # The bulk pool is the EPMC `resultType=core` fetch, so it wins for metadata the
                # curated row simply never had. It never overwrites a non-blank curated value.
                if value and not _blank(merged.get(col)) or (
                    col in ("mesh_headings", "pub_types", "keywords_author")
                    and _blank(merged.get(col)) in ("", "[]") and value not in ("", "[]")
                ):
                    merged[col] = value
            if not had_facets and any(_blank(merged.get(c)) not in ("", "[]") for c in
                                      ("mesh_headings", "pub_types", "keywords_author")):
                stats["backfilled_facets"] += 1

        pid = mint_landscape_pid(_blank(merged.get("pmcid")), _blank(merged.get("doi")),
                                 _blank(merged.get("pmid")))
        merged["pid"] = pid
        candidates.setdefault(pid, []).append(merged)

    normalised: list[dict] = []
    conflicts: list[dict] = []
    for pid, rows in sorted(candidates.items()):
        labels = {_blank(r.get("label")) for r in rows}
        if len(labels) > 1:
            # Two canonical rows for one paper carrying opposite labels. Measured: 5 papers, each
            # curated twice under two different canonical keys (one PMCID-keyed, one DOI-keyed) and
            # given opposite verdicts -- in one case seconds apart. The curator saw the same paper
            # twice without the queue revealing it was the same paper.
            #
            # Resolution rule (Gavin's call, 2026-09-03): **human judgement is retained.**
            #   1. `human_curated` beats `registry_confirmed`. A DOME Registry entry is a strong
            #      external claim, but a curator who read the paper and recorded a contrary verdict
            #      -- with a reason, e.g. "Not unsupervised ML - classical clustering" -- has
            #      overridden it deliberately. Provenance follows the winning row, so a paper
            #      resolved this way is published as `human_curated`, not `registry_confirmed`:
            #      claiming registry confirmation for a verdict that contradicts the registry
            #      would be false.
            #   2. Between two human decisions, the most recent real curation event wins -- the
            #      curator's latest judgement is their judgement.
            #
            # Every conflict is still written to `data/processed/curated_merge_conflicts.csv`, now
            # recording which row won and why, so the resolution is auditable rather than silent.
            resolved = max(rows, key=lambda r: _resolution_key(r, timestamps))
            stats["conflicts_resolved"] += 1
            for r in rows:
                won = r is resolved
                conflicts.append({
                    "pid": pid,
                    "record_id": _blank(r.get("record_id")),
                    "canonical_key": _blank(r.get("canonical_key")),
                    "pmcid": _blank(r.get("pmcid")),
                    "pmid": _blank(r.get("pmid")),
                    "doi": _blank(r.get("doi")),
                    "title": _blank(r.get("title"))[:300],
                    "label": _blank(r.get("label")),
                    "label_confidence": _blank(r.get("label_confidence")),
                    "cross_curate_final_label": _blank(r.get("cross_curate_final_label")),
                    "cross_curate_notes": _blank(r.get("cross_curate_notes")),
                    "notes": _blank(r.get("notes")),
                    "curation_timestamp": _blank(timestamps.get(_blank(r.get("record_id"))))
                                          or _blank(r.get("updated_at")),
                    "resolution": "PUBLISHED" if won else "superseded",
                    "resolution_reason": (
                        _resolution_reason(resolved, rows) if won
                        else "another row for this paper carried the retained human judgement"
                    ),
                })
            rows = [resolved]
            labels = {_blank(resolved.get("label"))}
        stats["deduped_away"] += len(rows) - 1
        winner = dict(max(rows, key=lambda r: sort_key(r, timestamps)))

        # Provenance is the STRONGEST claim in the group, not whichever row happened to win on
        # timestamp. Measured: picking by timestamp alone lost 162 of the 378 registry_confirmed
        # positives, because those papers also carry an agreeing human_curated row that was
        # written later. `registry_confirmed` is an externally verifiable DOME Registry entry --
        # a human review agreeing with it does not make it weaker. The labels are known equal
        # here (conflicts were held back above), so this only ever upgrades provenance.
        best_confidence = min(
            (_blank(r.get("label_confidence")) for r in rows),
            key=lambda c: _CONFIDENCE_RANK.get(c, 9),
        )
        if best_confidence != _blank(winner.get("label_confidence")):
            stats["provenance_upgraded"] += 1
            winner["label_confidence"] = best_confidence
        # ...and the rationale should name every source that agreed, not just the winner's.
        winner["sources"] = _merge_sources(rows)

        pmid = _blank(winner.get("pmid"))
        licence_entry = licensing.get(pmid)
        if licence_entry is not None:
            stats["licence_matched"] += 1
            licence, epmc_oa, checked = licence_entry[0], licence_entry[1], "True"
        else:
            licence, epmc_oa, checked = "", "", "False"

        citation = lookup_citation(
            citations, pmid=pmid, pmcid=_blank(winner.get("pmcid")), doi=_blank(winner.get("doi"))
        )
        if citation is not None:
            stats["citation_matched"] += 1
            count, updated, source = citation[0], citation[1], CITATION_SOURCE
        else:
            count, updated, source = "", "", ""

        rid = _blank(winner.get("record_id"))
        normalised.append({
            "pid": pid,
            "record_id": rid,
            "canonical_key": _blank(winner.get("canonical_key")),
            "pmid": pmid,
            "pmcid": _blank(winner.get("pmcid")),
            "doi": _blank(winner.get("doi")),
            "title": _blank(winner.get("title")),
            "abstract": _blank(winner.get("abstract")),
            "authors": _blank(winner.get("authors")),
            "year": _blank(winner.get("year")),
            "journal": _blank(winner.get("journal")),
            "citation_count": count,
            "citation_count_updated": updated,
            "citation_source": source,
            "mesh_headings": _blank(winner.get("mesh_headings")) or "[]",
            "pub_types": _blank(winner.get("pub_types")) or "[]",
            "keywords_author": _blank(winner.get("keywords_author")) or "[]",
            "is_open_access": _blank(winner.get("is_open_access")),
            "fulltext_available": _blank(winner.get("fulltext_available")),
            "abstract_source": _blank(winner.get("abstract_source")),
            "metadata_repair_sources": _blank(winner.get("metadata_repair_sources")),
            "license": licence,
            "license_checked": checked,
            "epmc_is_open_access": epmc_oa,
            "label": _blank(winner.get("label")),
            "label_confidence": _blank(winner.get("label_confidence")),
            "curation_rationale": build_rationale(winner),
            "curation_timestamp": _blank(timestamps.get(rid)) or _blank(winner.get("updated_at")),
            "curation_batch_id": batch_id,
        })
    return normalised, stats, conflicts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run(
    canonical_path: Path,
    bulk_path: Path,
    licensing_path: Path,
    citations_path: Path,
    out_jsonl: Path,
    out_csv: Path,
    out_report: Path,
    excluded_csv: Path,
    conflicts_csv: Path,
    batch_id: str,
    report_only: bool,
    strict: bool,
) -> None:
    print(f"build_curated_documents: batch_id {batch_id}\n")

    print(f"1. reading {canonical_path.name} ...")
    with canonical_path.open(newline="", encoding="utf-8", errors="replace") as f:
        canonical = list(csv.DictReader(f))
    print(f"   {len(canonical):,} canonical rows")

    existing_ids = load_existing_ids(canonical_path)
    print(f"   {len(existing_ids):,} distinct identifiers in the curated set")

    print(f"\n2. streaming {bulk_path.name} to recompute the exclusion (1.83GB, ~2 min) ...")
    by_id, n_excluded, excluded_rows = scan_bulk_pool(bulk_path, existing_ids)
    print(f"   {n_excluded:,} bulk rows were excluded as already curated "
          f"(expected {EXPECTED_EXCLUDED_BULK_ROWS:,})")

    in_pool = [r for r in canonical
               if any(_blank(r.get(c)) in by_id for c in ID_COLUMNS)]
    print(f"   {len(in_pool):,} canonical rows were in the bulk pool "
          f"(expected {EXPECTED_MATCHED_CANONICAL_ROWS:,})")
    print(f"   {len(canonical) - len(in_pool):,} were never in it "
          f"(heuristic_candidate sampler rows -- deliberately not published)")

    drift = []
    if n_excluded != EXPECTED_EXCLUDED_BULK_ROWS:
        drift.append(f"excluded bulk rows {n_excluded:,} != {EXPECTED_EXCLUDED_BULK_ROWS:,}")
    if len(in_pool) != EXPECTED_MATCHED_CANONICAL_ROWS:
        drift.append(f"matched canonical rows {len(in_pool):,} != {EXPECTED_MATCHED_CANONICAL_ROWS:,}")
    if drift:
        message = ("RECONCILIATION FAILED: " + "; ".join(drift) +
                   ". The premise of this merge is that these numbers are known and stable; if "
                   "they have moved, something upstream changed and needs checking before "
                   "anything is written. Re-run with --no-strict only after understanding why.")
        if strict:
            raise SystemExit(message)
        print(f"\n!! {message}\n")
    else:
        print("   reconciliation OK -- both figures match the verified values exactly")

    print("\n3. loading curation timestamps, licensing and citations ...")
    timestamps = load_curation_timestamps(CURATION_EVENT_LOGS)
    print(f"   {len(timestamps):,} records have a real dated curation event")
    licensing = load_licensing(licensing_path)
    print(f"   {len(licensing):,} pmids with a licence lookup")
    citations = load_citations(citations_path)
    print(f"   {len(citations):,} pids with a citation count")

    print("\n4. normalising, backfilling from the bulk pool, minting pids, deduping ...")
    normalised, stats, conflicts = normalise(
        in_pool, by_id, timestamps, licensing, citations, batch_id
    )
    print(f"   dropped 'skipped' label      : {stats['dropped_label']}")
    print(f"   dropped non-publishable conf : {stats['dropped_confidence']}")
    print(f"   no bulk twin found           : {stats['no_bulk_twin']}")
    print(f"   facets backfilled from bulk  : {stats['backfilled_facets']:,}")
    print(f"   duplicate canonical rows     : {stats['deduped_away']:,} collapsed by pid")
    print(f"   provenance upgraded on dedupe: {stats['provenance_upgraded']:,} "
          f"(registry_confirmed preserved over an agreeing human review)")
    print(f"   licence matched              : {stats['licence_matched']:,}")
    print(f"   citation matched             : {stats['citation_matched']:,}")
    print(f"   label conflicts resolved     : {stats['conflicts_resolved']:,} papers "
          f"(human judgement retained; {len(conflicts)} rows audited)")
    print(f"   -> {len(normalised):,} documents to build")

    if conflicts:
        print("\n   These papers were curated twice under different canonical keys and given")
        print("   opposite labels. Resolved by retaining human judgement -- human_curated over")
        print("   registry_confirmed, then most recent decision. Full audit trail in the")
        print("   conflicts CSV; PUBLISHED marks the verdict that survived.")
        for row in conflicts:
            mark = "->" if row["resolution"] == "PUBLISHED" else "  "
            print(f"     {mark} {row['pid'][:8]}  {row['label']:<9} {row['label_confidence']:<18} "
                  f"{row['cross_curate_notes'][:38]}")

    pids = [r["pid"] for r in normalised]
    if len(set(pids)) != len(pids):
        raise SystemExit("pid collision after dedupe -- this must never happen")

    empty_facets = sum(1 for r in normalised if r["mesh_headings"] == "[]"
                       and r["pub_types"] == "[]" and r["keywords_author"] == "[]")
    print(f"   documents still with no facets at all: {empty_facets:,}")

    from collections import Counter
    # Rows vs papers, spelled out: FINALISATION_ROADMAP.md's breakdown counts canonical ROWS
    # (2,870 negative + 3,093 positive human_curated + 378 positive registry_confirmed + 7
    # skipped + 8 undeterminable = 6,356). One paper can hold several rows -- 378
    # registry_confirmed rows are only 221 distinct papers, because a paper matched under both a
    # PMCID and a DOI key was recorded twice (AlphaFold 2 is exactly this). So a smaller
    # registry_confirmed figure below is the row->paper collapse, NOT lost provenance.
    row_counts = Counter((_blank(r.get("label")), _blank(r.get("label_confidence")))
                         for r in in_pool)
    print("\n   canonical ROWS in the pool (FINALISATION_ROADMAP section 2's figures):")
    for (label, conf), n in sorted(row_counts.items()):
        print(f"     {label:<16} {conf:<20} {n:,}")
    print("\n   documents (PAPERS) after pid dedupe -- label x provenance:")
    for (label, conf), n in sorted(Counter(
        (r["label"], r["label_confidence"]) for r in normalised
    ).items()):
        print(f"     {label:<16} {conf:<20} {n:,}")

    print("\n5. building documents ...")
    documents = []
    errors = []
    for row in tqdm(normalised, desc="documents", unit="doc"):
        try:
            documents.append(build_curated_document(row))
        except Exception as exc:  # noqa: BLE001 -- collect, report, never write a partial file
            errors.append({"pid": row["pid"], "record_id": row["record_id"], "error": repr(exc)})
    print(f"   {len(documents):,} built, {len(errors)} error(s)")
    for err in errors[:5]:
        print(f"     {err}")
    if errors:
        raise SystemExit("refusing to write with build errors -- fix them first")

    if report_only:
        print("\nbuild_curated_documents: --report-only, nothing written.")
        return

    write_atomic_csv(out_csv, normalised, NORMALISED_COLUMNS)
    print(f"\n6. wrote the auditable intermediate -> {out_csv}")

    write_atomic_csv(excluded_csv, excluded_rows,
                     ["pmcid", "pmid", "doi", "title", "journal", "year", "source_name"])
    print(f"   wrote the exclusion audit artifact -> {excluded_csv}")

    if conflicts:
        write_atomic_csv(conflicts_csv, conflicts, list(conflicts[0].keys()))
        print(f"   wrote {stats['conflicts_resolved']} resolved conflict(s), "
              f"{len(conflicts)} audited rows -> {conflicts_csv}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp, out_jsonl)
    print(f"   wrote {len(documents):,} documents -> {out_jsonl}")

    report = {
        "batch_id": batch_id,
        "inputs": {
            "canonical_dataset": {"path": str(canonical_path), "rows": len(canonical)},
            "bulk_candidates": {"path": str(bulk_path), "excluded_rows": n_excluded},
            "licensing": {"path": str(licensing_path), "pmids": len(licensing)},
            "citations": {"path": str(citations_path), "pids": len(citations)},
        },
        "reconciliation": {
            "excluded_bulk_rows": n_excluded,
            "expected_excluded_bulk_rows": EXPECTED_EXCLUDED_BULK_ROWS,
            "matched_canonical_rows": len(in_pool),
            "expected_matched_canonical_rows": EXPECTED_MATCHED_CANONICAL_ROWS,
            "passed": not drift,
        },
        "stats": stats,
        "documents_written": len(documents),
        "documents_with_no_facets": empty_facets,
        "label_conflicts_resolved": stats["conflicts_resolved"],
        "conflicts": conflicts,
        "errors": errors,
        "outputs": {
            "jsonl": {"path": str(out_jsonl), "sha256": sha256_file(out_jsonl)},
            "normalised_csv": {"path": str(out_csv), "sha256": sha256_file(out_csv)},
            "excluded_csv": {"path": str(excluded_csv), "sha256": sha256_file(excluded_csv)},
        },
    }
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"   wrote the conversion report -> {out_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--licensing", type=Path, default=DEFAULT_LICENSING)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT_REPORT)
    parser.add_argument("--excluded-csv", type=Path, default=DEFAULT_EXCLUDED_CSV)
    parser.add_argument("--conflicts-csv", type=Path, default=DEFAULT_CONFLICTS_CSV)
    parser.add_argument("--batch-id", default=None,
                        help="Defaults to curated_merge_<today>, stamped on every document.")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-strict", dest="strict", action="store_false",
                        help="Continue past a reconciliation mismatch. Understand why first.")
    args = parser.parse_args()

    from datetime import date
    batch_id = args.batch_id or f"curated_merge_{date.today().isoformat()}"
    run(args.canonical, args.bulk, args.licensing, args.citations, args.out_jsonl,
        args.out_csv, args.out_report, args.excluded_csv, args.conflicts_csv,
        batch_id, args.report_only, args.strict)


if __name__ == "__main__":
    main()
