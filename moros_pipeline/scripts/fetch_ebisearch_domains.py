"""Dumps whole the EBI Search domains whose entries cite publications, so papers can be matched
locally on PMID, PMCID or DOI. That reaches what the per-PMID route in `fetch_ebisearch_xrefs.py`
cannot: records known only by DOI or PMCID, and preprints, whose EBI Search `europepmc` entries
carry no cross-references. Raw staging only; `build_data_links.py` merges them.

By default only the accepted domains small enough to page through are dumped
(`ebisearch_resources.DUMP_DOMAINS`); `--all-citing` dumps every citing domain under the cap (the
2026-09-14 evaluation) and `--domain` names one. Each entry keeps its name/title and, where the
domain has one, `full_dataset_link`: the repository's own URL for the entry, which a link prefers
over a templated one.

    GET https://www.ebi.ac.uk/ebisearch/ws/rest?format=json
        the domain tree: each leaf's entry count, and every field whose `referenced domain` is
        europepmc (PUBMED, PMID, PUB_MED, EUROPE_PMC, PMC, PPR, DOI, ...)
    GET https://www.ebi.ac.uk/ebisearch/ws/rest/{domain}?query=domain_source:{domain}
        &size=100&start={n}&fields=id,name,<those fields>&format=json
        one page of entries; the pages of a domain are fetched in parallel

Measured 2026-09-14:
- 88 domains carry a retrievable field referencing europepmc (patent-number fields excluded: they
  never name a paper); 63 have at most 100,000 entries, 924,850 entries in all.
- `start` pages cleanly to the last entry below 100,000 (interpro7_family, 82,510 entries), but a
  page at start=100,000 answers 200 with no entries (chembl-document, 101,099). So the default cap
  is 100,000, and a page shorter than it should be is asked again twice and then fails the domain
  instead of truncating it.
- dome-registry (1,279 entries; `EUROPE_PMC` holds the PMID) took 13 calls in 0.2 s; biotools
  (34,230; PMID, PMCID, DOI) took 343 calls in 12 s.
- Asking for a field a domain lacks returns an empty list, not an error.
- Three index quirks, each the same on every read: physiome holds one entry with no id (the page
  at start=1,100 carries 100 entries, 99 with an id); ega lists EGAS00001006372 twice on one page,
  biostudies-arrayexpress E-MTAB-17162 on two pages, identical each time. An id-less entry is
  counted and left out; a repeat is written once after `id:"..."` confirms two indexed copies.

Each domain is written whole to `<out-dir>/<domain>.jsonl`, one line per entry with its fields
exactly as returned, through a temporary file, so a failed dump never leaves a partial file.
`_manifest.json` records per domain the entries indexed and written, the publication fields, EBI
Search's index dates and `fetched_at`: the provenance a later merge needs. A domain already dumped
is skipped unless it is older than `--max-age-days` or `--refresh` is given.

    python3 fetch_ebisearch_domains.py --list                  # the citing domains; * = dumped
    python3 fetch_ebisearch_domains.py --max-age-days 30       # the accepted domains, when stale
    python3 fetch_ebisearch_domains.py --refresh               # every accepted domain again
    python3 fetch_ebisearch_domains.py --domain dome-registry --domain biotools
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ebisearch_resources import DUMP_DOMAINS
from fetch_citations import _session

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR.parent / "output" / "ebisearch_domains"
MANIFEST_NAME = "_manifest.json"

BASE_URL = "https://www.ebi.ac.uk/ebisearch/ws/rest"
REFERENCED_DOMAIN = "europepmc"
# Field names that hold a paper's identifier. A domain qualifies only through a field EBI Search
# types as a europepmc reference; once it does, every field named here is taken whatever EBI Search
# says it references (dome-registry's `PMC` is typed as an NCBI reference).
PUBLICATION_FIELD_NAMES = frozenset({
    "PUBMED", "PMID", "PUB_MED", "PUBMED_ID", "MEDLINE", "PIMD", "EUROPE_PMC", "EUROPEPMC",
    "PMC", "PMCID", "PPR", "DOI",
})
EXCLUDED_FIELD_PREFIXES = ("PATENT",)
LABEL_FIELDS = ("name", "title")
LINK_FIELDS = ("full_dataset_link",)
PAGE_SIZE = 100
SHORT_PAGE_RETRIES = 2
DEFAULT_MAX_ENTRIES = 100_000
DEFAULT_MAX_WORKERS = 64
REQUEST_TIMEOUT = 30


class DumpError(RuntimeError):
    pass


def _options(field_info: dict) -> dict:
    return {o.get("name"): o.get("value") for o in field_info.get("options") or []}


def publication_fields(domain: dict) -> list[str]:
    """The retrievable fields holding a paper identifier, in the domain's own order; empty unless
    at least one of them references europepmc."""
    referencing = False
    fields: list[str] = []
    for info in domain.get("fieldInfos") or []:
        name = info.get("id") or ""
        opts = _options(info)
        if opts.get("retrievable") != "true" or name.upper().startswith(EXCLUDED_FIELD_PREFIXES):
            continue
        cites = (opts.get("referenced domain") or "").lower() == REFERENCED_DOMAIN
        referencing = referencing or cites
        if cites or name.upper() in PUBLICATION_FIELD_NAMES:
            fields.append(name)
    return fields if referencing else []


def leaf_domains(tree: dict) -> list[dict]:
    leaves: list[dict] = []

    def walk(domains: list[dict]) -> None:
        for domain in domains or []:
            if domain.get("subdomains"):
                walk(domain["subdomains"])
                continue
            info = {i.get("name"): i.get("value") for i in domain.get("indexInfos") or []}
            retrievable = {fi.get("id") for fi in domain.get("fieldInfos") or []
                           if _options(fi).get("retrievable") == "true"}
            try:
                entries = int(info.get("Number of entries") or 0)
            except ValueError:
                entries = 0
            leaves.append({
                "id": domain.get("id"),
                "entries": entries,
                "publication_fields": publication_fields(domain),
                "label_fields": [f for f in LABEL_FIELDS if f in retrievable],
                "link_fields": [f for f in LINK_FIELDS if f in retrievable],
                "index_updated": info.get("Update date"),
                "index_modified": info.get("Last modification date"),
            })

    walk(tree.get("domains") or [])
    return leaves


def select_domains(leaves: list[dict], wanted: list[str], max_entries: int,
                   accepted: frozenset[str] | set[str] | None = None) -> list[dict]:
    """The domains named (whatever their size), else every citing domain under the cap -- only the
    `accepted` ones when given -- smallest first."""
    citing = [leaf for leaf in leaves if leaf["publication_fields"]]
    if wanted:
        by_id = {leaf["id"]: leaf for leaf in citing}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            print(f"fetch_ebisearch_domains: not a citing domain: {', '.join(missing)}")
        return [by_id[w] for w in wanted if w in by_id]
    return sorted((leaf for leaf in citing if leaf["entries"] <= max_entries
                   and (accepted is None or leaf["id"] in accepted)),
                  key=lambda leaf: leaf["entries"])


def fields_param(leaf: dict) -> str:
    return ",".join(dict.fromkeys(["id", *leaf["label_fields"], *leaf.get("link_fields", []),
                                   *leaf["publication_fields"]]))


def reduce_page(payload: dict, domain: str) -> list[dict]:
    return [{"domain": domain, "id": e.get("id"), "fields": e.get("fields") or {}}
            for e in (payload or {}).get("entries") or [] if e.get("id")]


def fetch_page(session, domain: str, fields: str, start: int) -> tuple[int, int, list[dict]]:
    """hitCount, how many entries the page carried, and those of them with an id."""
    resp = session.get(f"{BASE_URL}/{domain}",
                       params={"query": f"domain_source:{domain}", "format": "json",
                               "size": PAGE_SIZE, "start": start, "fields": fields},
                       timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json() or {}
    return (int(payload.get("hitCount") or 0), len(payload.get("entries") or []),
            reduce_page(payload, domain))


def indexed_copies(session, domain: str, entry_id: str) -> int:
    resp = session.get(f"{BASE_URL}/{domain}",
                       params={"query": f'id:"{entry_id}"', "format": "json", "size": 0},
                       timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return int((resp.json() or {}).get("hitCount") or 0)


def dump_domain(session, leaf: dict, out_dir: Path, max_workers: int, fetched_at: str) -> dict:
    """Every page of one domain, checked complete before anything is written; returns the
    manifest entry. An entry with no id is counted and left out. An entry that comes back twice
    is written once, but only when EBI Search confirms it indexes two copies; otherwise the repeat
    means the pages moved while they were read, and the domain fails."""
    domain, fields = leaf["id"], fields_param(leaf)
    started = time.time()
    hit_count, carried, first = fetch_page(session, domain, fields, 0)
    starts = list(range(PAGE_SIZE, hit_count, PAGE_SIZE))
    pages = {0: (carried, first)}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        answers = pool.map(lambda start: fetch_page(session, domain, fields, start), starts)
        for start, (_, carried, entries) in zip(starts, answers):
            pages[start] = (carried, entries)
    calls = 1 + len(starts)
    for start in [0, *starts]:
        expected = min(PAGE_SIZE, hit_count - start)
        for _ in range(SHORT_PAGE_RETRIES):
            if pages[start][0] == expected:
                break
            _, carried, entries = fetch_page(session, domain, fields, start)
            pages[start] = (carried, entries)
            calls += 1
        if pages[start][0] != expected:
            raise DumpError(f"{domain}: page start={start} returned {pages[start][0]} entries, "
                            f"expected {expected} (hitCount {hit_count:,})")

    kept: list[dict] = []
    seen: dict[str, dict] = {}
    duplicates: list[str] = []
    for start in [0, *starts]:
        for entry in pages[start][1]:
            previous = seen.get(entry["id"])
            if previous is None:
                seen[entry["id"]] = entry
                kept.append(entry)
            elif previous == entry:
                duplicates.append(entry["id"])
            else:
                raise DumpError(f"{domain}: {entry['id']} came back twice with different fields "
                                "-- paging was not stable")
    for entry_id in dict.fromkeys(duplicates):
        calls += 1
        if indexed_copies(session, domain, entry_id) < 2:
            raise DumpError(f"{domain}: {entry_id} came back twice but is indexed once -- paging "
                            "was not stable")

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{domain}.jsonl"
    tmp = out_dir / f"{domain}.jsonl.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps({**entry, "fetched_at": fetched_at}, ensure_ascii=False) + "\n")
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "domain": domain,
        "entries_indexed": leaf["entries"],
        "hit_count": hit_count,
        "entries_written": len(kept),
        "entries_without_id": hit_count - len(kept) - len(duplicates),
        "duplicate_ids": duplicates,
        "publication_fields": leaf["publication_fields"],
        "fields_requested": fields,
        "calls": calls,
        "seconds": round(time.time() - started, 1),
        "index_updated": leaf["index_updated"],
        "index_modified": leaf["index_modified"],
        "fetched_at": fetched_at,
    }


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)


def is_fresh(manifest: dict, out_dir: Path, domain: str, max_age_days: int | None,
             now: datetime | None = None) -> bool:
    entry = manifest.get(domain)
    if not entry or not (out_dir / f"{domain}.jsonl").exists():
        return False
    if max_age_days is None:
        return True
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
    except (KeyError, ValueError):
        return False
    return fetched >= (now or datetime.now(timezone.utc)) - timedelta(days=max_age_days)


def run(out_dir: Path, wanted: list[str], max_entries: int, max_workers: int,
        max_age_days: int | None, refresh: bool, list_only: bool, tree_path: Path | None,
        accepted_only: bool = True) -> None:
    session = _session(max_workers)
    if tree_path is not None:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    else:
        resp = session.get(BASE_URL, params={"format": "json"}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        tree = resp.json()
    leaves = leaf_domains(tree)
    citing = sorted((leaf for leaf in leaves if leaf["publication_fields"]),
                    key=lambda leaf: leaf["entries"])
    selected = select_domains(leaves, wanted, max_entries,
                              DUMP_DOMAINS if accepted_only else None)
    print(f"fetch_ebisearch_domains: {len(leaves)} leaf domains, {len(citing)} cite publications; "
          f"{len(selected)} selected ({sum(leaf['entries'] for leaf in selected):,} entries)")
    if list_only:
        chosen = {leaf["id"] for leaf in selected}
        for leaf in citing:
            mark = "*" if leaf["id"] in chosen else " "
            print(f"  {mark} {leaf['entries']:>13,}  {leaf['id']:<34} "
                  f"{','.join(leaf['publication_fields'])}")
        return

    manifest = load_manifest(out_dir)
    fetched_at = datetime.now(timezone.utc).isoformat()
    n_calls = n_entries = n_failed = 0
    started = time.time()
    for leaf in selected:
        if not refresh and is_fresh(manifest, out_dir, leaf["id"], max_age_days):
            print(f"  {leaf['id']}: already dumped "
                  f"({manifest[leaf['id']]['entries_written']:,} entries), skipped")
            continue
        try:
            entry = dump_domain(session, leaf, out_dir, max_workers, fetched_at)
        except (DumpError, requests.RequestException, ValueError) as exc:
            n_failed += 1
            print(f"  {leaf['id']}: FAILED -- {exc!r}"[:300])
            continue
        manifest[leaf["id"]] = entry
        save_manifest(out_dir, manifest)
        n_calls += entry["calls"]
        n_entries += entry["entries_written"]
        rate = entry["calls"] / entry["seconds"] if entry["seconds"] else 0
        print(f"  {leaf['id']}: {entry['entries_written']:,} entries in {entry['calls']:,} calls, "
              f"{entry['seconds']}s ({rate:.1f} calls/s)")
    session.close()
    elapsed = time.time() - started
    print(f"\nfetch_ebisearch_domains: done -- {n_calls:,} calls in {elapsed:.1f}s "
          f"({n_calls / elapsed if elapsed else 0:.1f} calls/s), {n_entries:,} entries, "
          f"{n_failed} domains failed. Re-run the same command to retry failures.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--domain", action="append", default=[],
                        help="Dump this domain whatever its size (repeatable).")
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-age-days", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="Dump again even if fresh.")
    parser.add_argument("--list", action="store_true", help="List the citing domains and stop.")
    parser.add_argument("--tree", type=Path, default=None,
                        help="A saved copy of the domain tree instead of fetching it.")
    parser.add_argument("--all-citing", action="store_true",
                        help="Every citing domain under the cap, not only the accepted ones.")
    args = parser.parse_args()
    run(args.out_dir, args.domain, args.max_entries, args.max_workers, args.max_age_days,
        args.refresh, args.list, args.tree, accepted_only=not args.all_citing)


if __name__ == "__main__":
    main()
