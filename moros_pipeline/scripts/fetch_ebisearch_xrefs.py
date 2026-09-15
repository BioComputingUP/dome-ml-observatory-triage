"""Fetches EBI Search's cross-references for corpus papers: the entries in EBI's databases that cite
a paper -- a GEO series, an ENA project, a PDBe entry, a bio.tools tool, a DOME Registry review.
That is the database side of a paper's data links, which text mining the paper itself cannot see.
Raw staging files only; `build_data_links.py` merges them through `ebisearch_links.py`.

EBI Search keys its `europepmc` entries by PMID alone (a PMC or PPR id answers with no domains), so
both passes run over PMIDs:

    discover  GET https://www.ebi.ac.uk/ebisearch/ws/rest/europepmc/entry/{pmid}/xref?format=json
              which domains cite the paper, and with how many entries. One PMID per call: a comma
              list without a target domain is HTTP 400. An unknown PMID answers {"domains": []}.
    detail    GET .../europepmc/entry/{pmid,pmid,...}/xref/{domain}?fields=id,name&size=100
              the citing entries, up to 100 PMIDs per call (101 is HTTP 400). An unknown PMID is
              left out of the answer.

Measured 2026-09-14:
- Discovery ran at 79.7 calls/s at 128 workers (3,000-PMID sample, no errors) and over every
  positive PMID at 256 workers: 316,524 calls in 34 min, 154 calls/s, 49 ConnectTimeouts (0.015%),
  all but one recovered by a re-run. 29% of papers had a cross-reference, 96% of those only the
  BioStudies supplementary entry.
- `size` must be sent, and as 100. It applies per entry, and without it (or with a small value)
  the answer carries ONE reference whatever the entry's referenceCount (pdbekb for 33024307: 1 of
  5; intact-interactions for 37398436: 1 of 94). Every entry is therefore checked against
  min(referenceCount, 100), and a short one is asked again on its own.
- Counts run to 125,077 (uniprot for PMID 11237011). `start` in steps of 100 pages one entry
  stably; `--max-refs` (default 100, one page) caps what is kept, and `reference_count` is always
  recorded, so a capped record says so.
- Asking for a field a domain lacks returns an empty list, not an error.

`discover` covers positives: the input's `classification` column, or for a staged batch (which has
none) `--classification-events`. `detail` asks the accepted domains too large to dump
(`ebisearch_resources.XREF_DOMAINS`); `--domain` names others and `--all-domains` asks every domain
discovery listed (the evaluation). Domains dumped whole by `fetch_ebisearch_domains.py` are skipped
(the dump already holds every entry with its publication identifiers); `--include-dumped` asks them
anyway, to cross-check.

A PMID EBI Search errors on every time (18575676, 2026-09-14) would withhold its paper's data links
forever, since the merge waits for discovery. `discover --record-failures` writes a `failed` record
for a PMID that still fails after the session's retries; the merge counts it as answered with no
cross-references, and `--max-age-days` asks it again later.

Records are raw and append-only. The latest line for a PMID (discover) or a domain and PMID
(detail) wins, and an incomplete detail record is asked again on the next run.

    python3 fetch_ebisearch_xrefs.py discover --limit 3000        # sample: rate, errors, domains
    python3 fetch_ebisearch_xrefs.py discover --max-workers 256   # every positive PMID
    python3 fetch_ebisearch_xrefs.py discover --record-failures   # a re-run: record what still fails
    python3 fetch_ebisearch_xrefs.py detail --limit 3000
    python3 fetch_ebisearch_xrefs.py detail
    python3 fetch_ebisearch_xrefs.py discover --input ../output/incoming_new.csv \
        --classification-events ../output/incoming_new_classification_events.csv   # a batch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from classification_events import latest_verdicts
from ebisearch_resources import XREF_DOMAINS
from fetch_annotations import load_already_fetched
from fetch_citations import _session

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR.parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "corpus_keys.csv"
DEFAULT_DISCOVERY = OUTPUT_DIR / "ebisearch_xref_discovery.jsonl"
DEFAULT_DETAIL = OUTPUT_DIR / "ebisearch_xref_detail.jsonl"
DEFAULT_DUMPS_DIR = OUTPUT_DIR / "ebisearch_domains"

BASE_URL = "https://www.ebi.ac.uk/ebisearch/ws/rest/europepmc/entry"
SOURCE = "MED"                # EBI Search's europepmc entries are PMIDs
IDS_PER_CALL = 100            # hard limit
PAGE_SIZE = 100               # hard limit on `size`, per entry
DEFAULT_MAX_REFS = PAGE_SIZE
SHORT_RETRIES = 2
DISCOVERY_TIMEOUT = 15
DETAIL_TIMEOUT = 60
DEFAULT_DISCOVERY_WORKERS = 128
DEFAULT_DETAIL_WORKERS = 64


def load_targets(input_path: Path, classification: str | None,
                 verdicts: dict[str, str] | None = None) -> list[str]:
    """PMIDs of the rows with `classification` (every row when None), each once, in file order.
    A row's verdict is its `classification` column, or `verdicts[pid]` when a batch's event log is
    given (a staged CSV carries no column). Asking for a classification the input cannot supply
    stops, rather than silently taking every row."""
    seen: set[str] = set()
    pmids: list[str] = []
    n = 0
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if (classification is not None and verdicts is None
                and "classification" not in (reader.fieldnames or [])):
            raise SystemExit(
                f"fetch_ebisearch_xrefs: {input_path.name} has no classification column -- pass "
                f"--classification-events <the batch's events CSV>, or --classification all")
        for row in reader:
            n += 1
            if classification is not None:
                verdict = (verdicts.get((row.get("pid") or "").strip()) if verdicts is not None
                           else (row.get("classification") or "").strip())
                if verdict != classification:
                    continue
            pmid = (row.get("pmid") or "").strip()
            if pmid.isdigit() and pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)
    print(f"fetch_ebisearch_xrefs: {input_path.name}: {n:,} rows -> {len(pmids):,} PMIDs "
          f"({classification or 'every classification'})")
    return pmids


def fetch_discovery(session, pmid: str, fetched_at: str) -> dict:
    resp = session.get(f"{BASE_URL}/{pmid}/xref", params={"format": "json"},
                       timeout=DISCOVERY_TIMEOUT)
    resp.raise_for_status()
    return {"source": SOURCE, "id": pmid, "fetched_at": fetched_at,
            "http_status": resp.status_code, "domains": (resp.json() or {}).get("domains") or []}


def discover_or_fail(session, pmid: str, fetched_at: str) -> dict:
    """Discovery for one PMID, or a `failed` record when it still fails after the session's own
    retries. Written only under `--record-failures`: a transient failure must be asked again."""
    try:
        return fetch_discovery(session, pmid, fetched_at)
    except Exception as exc:  # noqa: BLE001 -- recorded, by request, instead of raised
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"source": SOURCE, "id": pmid, "fetched_at": fetched_at, "http_status": status,
                "failed": True, "error": repr(exc)[:200], "domains": []}


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def latest_discovery(path: Path) -> dict[str, dict]:
    return {str(rec["id"]): rec for rec in _read_jsonl(path) if rec.get("id")}


def load_done_pairs(path: Path, max_age_days: int | None) -> set[tuple[str, str]]:
    """(domain, pmid) pairs whose latest detail record is complete (and younger than max_age_days,
    when given)."""
    latest = {(rec.get("domain"), str(rec.get("id"))): rec for rec in _read_jsonl(path)}
    cutoff = None
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    done: set[tuple[str, str]] = set()
    for pair, rec in latest.items():
        if not rec.get("complete"):
            continue
        if cutoff is not None:
            try:
                if datetime.fromisoformat(rec["fetched_at"]) < cutoff:
                    continue
            except (KeyError, ValueError):
                continue
        done.add(pair)
    return done


def detail_pairs(discovery: dict[str, dict], skip_domains: set[str],
                 only_domains: set[str] | None = None) -> list[tuple[str, str]]:
    pairs = []
    for pmid, rec in discovery.items():
        for d in rec.get("domains") or []:
            domain = d.get("id")
            if not domain or domain in skip_domains:
                continue
            if only_domains and domain not in only_domains:
                continue
            if int(d.get("referenceEntryCount") or 0) > 0:
                pairs.append((domain, pmid))
    return sorted(pairs)


def detail_domains(only_domains: list[str] | None, all_domains: bool) -> set[str] | None:
    """The domains `detail` asks: those named, else the accepted domains too large to dump, else
    (with `--all-domains`, for evaluation) every domain discovery listed."""
    if only_domains:
        return set(only_domains)
    return None if all_domains else set(XREF_DOMAINS)


def batches(pairs: Iterable[tuple[str, str]],
            size: int = IDS_PER_CALL) -> list[tuple[str, list[str]]]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for domain, pmid in pairs:
        by_domain[domain].append(pmid)
    return [(domain, pmids[i:i + size]) for domain, pmids in sorted(by_domain.items())
            for i in range(0, len(pmids), size)]


def reference_record(ref: dict) -> dict:
    rec = {"id": ref.get("id"), "fields": ref.get("fields") or {}}
    if ref.get("acc"):
        rec["acc"] = ref["acc"]
    return rec


def _entries(session, domain: str, pmids: list[str], start: int = 0) -> dict[str, dict]:
    params = {"format": "json", "fields": "id,name", "size": PAGE_SIZE}
    if start:
        params["start"] = start
    resp = session.get(f"{BASE_URL}/{','.join(pmids)}/xref/{domain}", params=params,
                       timeout=DETAIL_TIMEOUT)
    resp.raise_for_status()
    return {str(e.get("id")): e for e in (resp.json() or {}).get("entries") or []}


def _is_short(entry: dict) -> bool:
    expected = min(int(entry.get("referenceCount") or 0), PAGE_SIZE)
    return len(entry.get("references") or []) < expected


def fetch_detail_batch(session, domain: str, pmids: list[str], fetched_at: str,
                       max_refs: int = DEFAULT_MAX_REFS) -> tuple[list[dict], int]:
    """One call for the batch; then, per PMID, a re-ask on its own while its answer is short, and
    further pages until min(referenceCount, max_refs) references are held. Returns the records
    and the number of calls made."""
    answer = _entries(session, domain, pmids)
    calls = 1
    records = []
    for pmid in pmids:
        base = {"source": SOURCE, "id": pmid, "domain": domain, "fetched_at": fetched_at}
        entry = answer.get(pmid)
        for _ in range(SHORT_RETRIES):
            if entry is None or not _is_short(entry):
                break
            entry = _entries(session, domain, [pmid]).get(pmid)
            calls += 1
        if entry is None:
            records.append({**base, "reference_count": None, "truncated": False,
                            "complete": False, "references": []})
            continue
        count = int(entry.get("referenceCount") or 0)
        refs = list(entry.get("references") or [])
        target = min(count, max_refs)
        while not _is_short(entry) and len(refs) < target:
            page = _entries(session, domain, [pmid], start=len(refs)).get(pmid)
            calls += 1
            got = (page or {}).get("references") or []
            if not got:
                break
            refs.extend(got)
        refs = refs[:max_refs]
        records.append({**base, "reference_count": count, "truncated": count > max_refs,
                        "complete": len(refs) >= target,
                        "references": [reference_record(r) for r in refs]})
    return records, calls


def _sample(items: list, limit: int | None, label: str) -> list:
    if limit is None:
        return items
    picked = sorted(random.Random(42).sample(items, min(limit, len(items))))
    print(f"fetch_ebisearch_xrefs: --limit {limit} -> {len(picked):,} {label} this run")
    return picked


def _run_pool(jobs: list, call: Callable, output_path: Path, desc: str, max_workers: int,
              tally: Callable[[dict], None]) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_jobs = n_calls = n_failed = n_written = 0
    started = time.time()
    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(call, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="job"):
                n_jobs += 1
                try:
                    records, calls = future.result()
                except Exception as exc:  # noqa: BLE001 -- a re-run asks again
                    n_failed += 1
                    n_calls += 1
                    if n_failed <= 20:
                        tqdm.write(f"  {str(futures[future])[:80]} FAILED: {exc!r}"[:200])
                    continue
                n_calls += calls
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tally(rec)
                n_written += len(records)
                if n_jobs % 500 == 0:
                    f.flush()
    return {"jobs": n_jobs, "calls": n_calls, "failed": n_failed, "written": n_written,
            "elapsed": time.time() - started}


def _print_rate(name: str, stats: dict) -> None:
    elapsed, jobs = stats["elapsed"], stats["jobs"]
    print(f"\nfetch_ebisearch_xrefs {name}: done -- {stats['calls']:,} calls in {elapsed:.1f}s "
          f"({stats['calls'] / elapsed if elapsed else 0:.1f} calls/s), {stats['failed']} of "
          f"{jobs:,} jobs failed ({stats['failed'] / jobs * 100 if jobs else 0:.1f}%). "
          f"{stats['written']:,} records written. Re-run the same command to retry failures.")


def run_discover(input_path: Path, output_path: Path, classification: str | None,
                 max_workers: int, limit: int | None, max_age_days: int | None,
                 verdicts: dict[str, str] | None = None, record_failures: bool = False) -> None:
    pmids = load_targets(input_path, classification, verdicts)
    already = load_already_fetched(output_path, max_age_days)
    remaining = [p for p in pmids if (SOURCE, p) not in already]
    print(f"fetch_ebisearch_xrefs discover: {len(already):,} already fetched | "
          f"{len(remaining):,} remaining")
    remaining = _sample(remaining, limit, "PMIDs")
    if not remaining:
        print("fetch_ebisearch_xrefs discover: nothing to do.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    papers: Counter = Counter()
    entries: Counter = Counter()
    cited = failed = 0

    def tally(rec: dict) -> None:
        nonlocal cited, failed
        failed += bool(rec.get("failed"))
        hits = [d for d in rec["domains"] if int(d.get("referenceEntryCount") or 0) > 0]
        cited += bool(hits)
        for d in hits:
            papers[d["id"]] += 1
            entries[d["id"]] += int(d["referenceEntryCount"])

    fetch = discover_or_fail if record_failures else fetch_discovery
    stats = _run_pool(remaining, lambda p: ([fetch(session, p, fetched_at)], 1),
                      output_path, "discover", max_workers, tally)
    session.close()
    _print_rate("discover", stats)
    print(f"  papers cited by any domain: {cited:,} of {stats['written']:,}"
          + (f"; {failed:,} recorded as failed" if failed else ""))
    for domain, n in papers.most_common(40):
        print(f"    {domain:<34} {n:>9,} papers {entries[domain]:>11,} entries")


def run_detail(discovery_path: Path, output_path: Path, dumps_dir: Path, include_dumped: bool,
               only_domains: list[str] | None, max_workers: int, max_refs: int,
               limit: int | None, max_age_days: int | None, all_domains: bool = False) -> None:
    discovery = latest_discovery(discovery_path)
    dumped = set()
    if not include_dumped and dumps_dir.exists():
        dumped = {p.stem for p in dumps_dir.glob("*.jsonl")}
    pairs = detail_pairs(discovery, dumped, detail_domains(only_domains, all_domains))
    done = load_done_pairs(output_path, max_age_days)
    remaining = [pair for pair in pairs if pair not in done]
    print(f"fetch_ebisearch_xrefs detail: {len(discovery):,} discovery records -> {len(pairs):,} "
          f"(domain, PMID) pairs, {len(dumped)} dumped domains skipped | {len(done):,} done | "
          f"{len(remaining):,} remaining")
    remaining = _sample(remaining, limit, "pairs")
    jobs = batches(remaining)
    if not jobs:
        print("fetch_ebisearch_xrefs detail: nothing to do.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    refs: Counter = Counter()
    counts = Counter()

    def tally(rec: dict) -> None:
        refs[rec["domain"]] += len(rec["references"])
        counts["incomplete"] += not rec["complete"]
        counts["truncated"] += rec["truncated"]

    stats = _run_pool(jobs, lambda job: fetch_detail_batch(session, job[0], job[1], fetched_at,
                                                           max_refs),
                      output_path, "detail", max_workers, tally)
    session.close()
    _print_rate("detail", stats)
    print(f"  references kept: {sum(refs.values()):,}; records incomplete: "
          f"{counts['incomplete']:,} (asked again next run); capped at --max-refs {max_refs}: "
          f"{counts['truncated']:,}")
    for domain, n in refs.most_common(40):
        print(f"    {domain:<34} {n:>11,} references")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Which domains cite each PMID.")
    discover.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    discover.add_argument("--output", type=Path, default=DEFAULT_DISCOVERY)
    discover.add_argument("--classification", default="positive",
                          help="Rows to take from the input; 'all' for every row.")
    discover.add_argument("--classification-events", type=Path, default=None,
                          help="A staged batch's classification event log, for an input with no "
                               "classification column.")
    discover.add_argument("--record-failures", action="store_true",
                          help="Write a `failed` record for a PMID that still fails after retries.")
    discover.add_argument("--max-workers", type=int, default=DEFAULT_DISCOVERY_WORKERS)
    discover.add_argument("--limit", type=int, default=None)
    discover.add_argument("--max-age-days", type=int, default=None)

    detail = sub.add_parser("detail", help="The citing entries, per domain, 100 PMIDs a call.")
    detail.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    detail.add_argument("--output", type=Path, default=DEFAULT_DETAIL)
    detail.add_argument("--dumps-dir", type=Path, default=DEFAULT_DUMPS_DIR)
    detail.add_argument("--include-dumped", action="store_true",
                        help="Ask domains already dumped whole too (a cross-check).")
    detail.add_argument("--domain", action="append", default=None,
                        help="Only this domain (repeatable).")
    detail.add_argument("--all-domains", action="store_true",
                        help="Every domain discovery listed, not only the accepted ones.")
    detail.add_argument("--max-workers", type=int, default=DEFAULT_DETAIL_WORKERS)
    detail.add_argument("--max-refs", type=int, default=DEFAULT_MAX_REFS,
                        help="References kept per PMID and domain; above 100 pages with `start`.")
    detail.add_argument("--limit", type=int, default=None)
    detail.add_argument("--max-age-days", type=int, default=None)

    args = parser.parse_args()
    if args.command == "discover":
        verdicts = latest_verdicts(args.classification_events) if args.classification_events else None
        run_discover(args.input, args.output,
                     None if args.classification == "all" else args.classification,
                     args.max_workers, args.limit, args.max_age_days, verdicts,
                     args.record_failures)
    else:
        run_detail(args.discovery, args.output, args.dumps_dir, args.include_dumped, args.domain,
                   args.max_workers, args.max_refs, args.limit, args.max_age_days,
                   args.all_domains)


if __name__ == "__main__":
    main()
