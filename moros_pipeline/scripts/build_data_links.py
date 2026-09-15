"""Merges every Europe PMC route into one `data_links` block per document, and writes the two
staging files the field loads read.

Inputs, all keyed by the record's Europe PMC identity:

- `epmc_metadata.csv` (`fetch_epmc_metadata.py`): identity, preprint server, the summary flags;
- `epmc_annotations.jsonl` (`fetch_annotations.py`): text-mined accessions, the primary link source;
- `epmc_datalinks.jsonl` (`fetch_datalinks.py`): the Scholix residual (cross-references, data
  citations, external links);
- `epmc_textmined_bulk.jsonl` (`import_textmined_bulk.py`, optional): the FTP dump, as a
  cross-check or a fallback for the annotations API;
- and one link that needs no fetch: a record with supplementary files in PMC has a BioStudies
  entry `S-EPMC<digits>` (derived from `has_suppl` and the pmcid; verified live 2026-09-14);
- and, for positives (schema v1.5.0), EBI Search's database-side links: `ebisearch_xref_*.jsonl`
  (`fetch_ebisearch_xrefs.py`) and `ebisearch_domains/` (`fetch_ebisearch_domains.py`), keyed on OUR
  pmid / pmcid / doi and restricted to the accept list in `ebisearch_resources.py`
  (`ebisearch_links.py` reads them). Every link, from every route, is filed under its home resource
  before the dedupe (`ebisearch_resources.canonicalise_link`: an ArrayExpress `E-GEOD-n` is GEO
  `GSEn`, a versioned dbGaP study is the study, a PXD goes to the partner hosting it), so two routes
  naming one accession make one link, and the resource's `routes` lists both.

Outputs:

- `pid_preprints.csv` -> `load_fields.py --mode preprints` (`pid, epmc_source, epmc_id,
  preprint_server`), one row per document Europe PMC answered for;
- `pid_data_links.csv` -> `load_fields.py --mode data_links` (`pid, has_data, data_links_tags,
  accession_types, db_cross_references, data_links_json`). The summary columns are always written.
  The JSON cell -- `fetched_at, sources, link_count, truncated, resources[], links[]` -- is written
  only once every route the record was *targeted* for has answered and every DOI it links to has
  been confirmed at doi.org, so `data_links.fetched_at` stays null (never fetched) rather than
  claiming completeness for a half-fetched record.
- `pid_identifiers.csv` -> `load_fields.py --mode identifiers` (`pid, dome_registry`), written when
  the EBI Search route runs: the DOME Registry entry naming the paper, `""` for an in-scope paper
  with a PMID or PMCID and none, no row for a paper that cannot be looked up.

**This is the only place a link identifier is chosen** (`link_identifiers.py` defines clean).
Europe PMC's text-mined strings carry the punctuation, quotes and words around them; stored
verbatim they made links that 404. Per link: a DOI is repaired into its candidates
(`link_identifiers.doi_candidates`) and the first one registered at doi.org wins (unregistered:
dropped; not yet confirmed: the record is withheld); any other accession is taken as mined if
clean, else from Europe PMC's resolver URL, else normalised or cleaned, else dropped. The DOI
confirmations run inside the build, before any document is assembled, and are cached in
`output/doi_handles.csv` (negatives re-asked after `--handle-max-age-days`). Every assembled block
is then checked with `malformed_links()`; a failure raises before either output file is replaced.

The join key is derived from OUR corpus row (`fetch_epmc_metadata.metadata_key`), never from
Europe PMC's echo, exactly as `join_citations.py` insists. Links are deduplicated on
`(resource, id)`, so overlapping routes are safe; `datalinks_resources.slug()` decides the
resource, and everything it cannot map is tabulated by `--report-only` so the table can grow.
Caps: 50 links per resource, 300 per document; the true totals survive in `count` / `link_count`.

Memory is bounded by `--shards` (default 8): documents are split by a stable hash of their pid, and
for each shard only the metadata rows and JSONL records that shard needs are held -- a JSONL line's
identity is read from its prefix, so the other shards' records are never parsed. The whole corpus
therefore builds on a machine with a few GB free, at the cost of reading each input once per shard.

    python3 build_data_links.py --report-only            # coverage, resource mix, repairs, drops
    python3 build_data_links.py                          # -> ../output/pid_preprints.csv, pid_data_links.csv
    python3 build_data_links.py --keys ../output/incoming_new.csv --metadata ../output/incoming_new.csv \
        --classification-events ../output/incoming_new_classification_events.csv --shards 1 \
        --out-identifiers ../output/incoming_new_pid_identifiers.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from classification_events import latest_verdicts
from datalinks_resources import describe, doi_prefix, is_reference_section, scheme_from_uri, slug
from ebisearch_links import EbiRoute, paper_keys
from ebisearch_resources import browse_url, canonicalise_link, resolve_pxd_hosts
from fetch_citations import _session
from fetch_datalinks import wanted as datalinks_targeted
from fetch_epmc_metadata import metadata_key
from link_identifiers import (
    canonical_url,
    clean,
    doi_candidates,
    malformed_links,
    normalise_prefixed,
    problems,
    resolver_local_id,
)

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
OUTPUT_DIR = FOLDER_DIR / "output"
DEFAULT_KEYS = OUTPUT_DIR / "corpus_keys.csv"
DEFAULT_METADATA = OUTPUT_DIR / "epmc_metadata.csv"
DEFAULT_ANNOTATIONS = OUTPUT_DIR / "epmc_annotations.jsonl"
DEFAULT_DATALINKS = OUTPUT_DIR / "epmc_datalinks.jsonl"
DEFAULT_BULK = OUTPUT_DIR / "epmc_textmined_bulk.jsonl"
DEFAULT_OUT_PREPRINTS = OUTPUT_DIR / "pid_preprints.csv"
DEFAULT_OUT_DATA_LINKS = OUTPUT_DIR / "pid_data_links.csv"
DEFAULT_HANDLES = OUTPUT_DIR / "doi_handles.csv"
DEFAULT_EBISEARCH_DISCOVERY = OUTPUT_DIR / "ebisearch_xref_discovery.jsonl"
DEFAULT_EBISEARCH_DETAIL = OUTPUT_DIR / "ebisearch_xref_detail.jsonl"
DEFAULT_EBISEARCH_DOMAINS = OUTPUT_DIR / "ebisearch_domains"
DEFAULT_OUT_IDENTIFIERS = OUTPUT_DIR / "pid_identifiers.csv"

MAX_LINKS_PER_RESOURCE = 50
MAX_LINKS_PER_DOCUMENT = 300

SOURCE_SEARCH = "epmc_search"           # only the search record was consulted (nothing to fetch)
SOURCE_ANNOTATIONS = "epmc_annotations"
SOURCE_DATALINKS = "epmc_datalinks"
SOURCE_BULK = "epmc_textmined_bulk"
SOURCE_DERIVED = "derived"
SOURCE_EBISEARCH = "ebisearch"        # EBI Search's database-side links (v1.5.0, positives)

PREPRINT_COLUMNS = ["pid", "epmc_source", "epmc_id", "preprint_server"]
DATA_LINKS_COLUMNS = ["pid", "has_data", "data_links_tags", "accession_types",
                      "db_cross_references", "data_links_json"]
IDENTIFIER_COLUMNS = ["pid", "dome_registry"]
# v1.5.0 added matched_by / source_domain: which of our identifiers an EBI Search entry named, and
# the domain that asserted the link (both null for the Europe PMC routes).
LINK_KEYS = ("resource", "id", "url", "title", "obtained_by", "relationship", "section", "frequency",
             "matched_by", "source_domain")
# v1.5.0 added routes (every route that found a link to the resource) and browse_url.
RESOURCE_KEYS = ("resource", "label", "category", "id_scheme", "publisher", "obtained_by", "count",
                 "routes", "browse_url")

# doi.org's handle API: responseCode 1 = registered, 100 = not registered (measured 2026-09-14:
# 464 lookups/s at 64 workers). Values-not-found (200) still means the handle exists.
HANDLE_API = "https://doi.org/api/handles/"
HANDLE_COLUMNS = ["doi", "exists", "checked_at"]
HANDLE_TIMEOUT = 15
DEFAULT_HANDLE_WORKERS = 64
DEFAULT_HANDLE_MAX_AGE_DAYS = 30

# A link whose DOI has not been confirmed at doi.org yet. Its record is withheld, never written
# half-verified.
PENDING = object()


class Stats:
    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.resources: Counter = Counter()          # documents carrying each resource
        self.unmapped_scheme: Counter = Counter()
        self.unmapped_doi_prefix: Counter = Counter()
        self.reference_dois: int = 0
        self.repaired: Counter = Counter()           # links whose stored id differs from the mined one
        self.dropped: Counter = Counter()            # (reason, resource) -> links not stored
        self.from_resolver: int = 0
        self.doi_pending: int = 0
        self.ebisearch_resources: Counter = Counter()     # documents where EBI Search added a link to each
        self.ebisearch_confirmed: Counter = Counter()     # documents where EBI Search found a link already held
        self.ebisearch_rejected: Counter = Counter()      # domains EBI Search listed that are not accepted
        self.ebisearch_unclassified: Counter = Counter()  # (domain, field) values of no known shape


# ---------------------------------------------------------------------------
# Choosing an identifier
# ---------------------------------------------------------------------------


def _norm_scheme(scheme: str | None) -> str:
    return (scheme or "").strip().lower()


def _choose_doi(candidates: list[str], handles: dict[str, bool]) -> str | object | None:
    """The first candidate doi.org confirms; PENDING when a candidate has no verdict yet (never
    skip past it -- it may be the right one); None when every candidate is unregistered."""
    for candidate in candidates:
        verdict = handles.get(candidate.lower())
        if verdict is True:
            return candidate
        if verdict is None:
            return PENDING
    return None


def _choose_other(exact: str, uri: str | None, resource: str) -> tuple[str, str] | None:
    """(identifier, how) for a non-DOI accession, or None when nothing clean can be recovered.
    `how` is exact | resolver | prefixed | cleaned."""
    if not problems(exact, resource):
        return exact, "exact"
    prefixed = normalise_prefixed(exact)
    local = resolver_local_id(uri)
    if local:
        local = clean(local)
        if local and not problems(local, resource):
            # `ORPHA 401777` with `.../orphanet:401777` keeps its prefix: `ORPHA:401777`.
            if (prefixed != exact and prefixed.endswith(":" + local)
                    and not problems(prefixed, resource)):
                return prefixed, "resolver"
            return local, "resolver"
    if prefixed != exact and not problems(prefixed, resource):
        return prefixed, "prefixed"
    cleaned = clean(exact)
    if cleaned and not problems(cleaned, resource):
        return cleaned, "cleaned"
    return None


def _is_doi_annotation(annotation: dict) -> bool:
    scheme = annotation.get("sub_type") or scheme_from_uri(annotation.get("uri"))
    return _norm_scheme(scheme) == "doi"


def annotation_doi_candidates(annotation: dict) -> list[str]:
    """The DOIs a text-mined annotation needs confirmed at doi.org. None for a reference-list
    citation or a DOI no data repository owns: those never become links, so they are never looked
    up (there are ~1M of them)."""
    exact = (annotation.get("exact") or "").strip()
    if not exact or not _is_doi_annotation(annotation) or is_reference_section(annotation.get("section")):
        return []
    if slug("DOI", None, exact) is None:
        return []
    return [c for c in doi_candidates(exact) if slug("DOI", None, c) is not None]


def datalink_doi_candidates(link: dict) -> list[str]:
    ident = (link.get("id") or "").strip()
    if not ident or _norm_scheme(link.get("id_scheme")) != "doi":
        return []
    return doi_candidates(ident)


# ---------------------------------------------------------------------------
# One link from each route
# ---------------------------------------------------------------------------


def link_from_annotation(annotation: dict, stats: Stats,
                         handles: dict[str, bool] | None = None) -> dict | object | None:
    """A text-mined accession -> a link; None for a literature DOI, an unmappable scheme, or an
    identifier nothing clean can be recovered from; PENDING while its DOI awaits doi.org."""
    exact = (annotation.get("exact") or "").strip()
    if not exact:
        return None
    handles = handles or {}
    scheme = annotation.get("sub_type") or scheme_from_uri(annotation.get("uri"))
    uri = annotation.get("uri")
    if _norm_scheme(scheme) == "doi":
        if is_reference_section(annotation.get("section")):
            stats.reference_dois += 1
            return None
        guess = slug("DOI", None, exact)
        if guess is None:
            stats.unmapped_doi_prefix[doi_prefix(exact) or "?"] += 1
            return None
        candidates = annotation_doi_candidates(annotation)
        if not candidates:
            stats.dropped[("malformed DOI", guess)] += 1
            return None
        chosen = _choose_doi(candidates, handles)
        if chosen is PENDING:
            stats.doi_pending += 1
            return PENDING
        if chosen is None:
            stats.dropped[("DOI not registered", guess)] += 1
            return None
        link_id = chosen
        resource = slug("DOI", None, link_id)
    else:
        resource = slug(scheme)
        if resource is None:
            stats.unmapped_scheme[scheme or "?"] += 1
            return None
        choice = _choose_other(exact, uri, resource)
        if choice is None:
            stats.dropped[("malformed", resource)] += 1
            return None
        link_id, how = choice
        if how == "resolver":
            stats.from_resolver += 1
    if link_id != exact:
        stats.repaired[resource] += 1
    provider = annotation.get("provider") or "Europe PMC"
    return {
        "resource": resource,
        "id": link_id,
        "url": canonical_url(resource, link_id, uri),
        "title": None,
        "obtained_by": "tm_supplementary" if provider.lower() == "biostudies" else "tm_accession",
        "relationship": "References",
        "section": annotation.get("section"),
        "frequency": annotation.get("frequency"),
        "_id_scheme": scheme,
        "_publisher": provider,
        "_category": None,
    }


def link_from_datalink(link: dict, stats: Stats,
                       handles: dict[str, bool] | None = None) -> dict | object | None:
    """A Scholix link -> a link. The publisher names the resource for a DOI whose prefix is not in
    the table (Europe PMC already vetted it as data), so nothing Scholix vouches for is lost."""
    ident = (link.get("id") or "").strip()
    if not ident:
        return None
    handles = handles or {}
    scheme, publisher = link.get("id_scheme"), link.get("publisher")
    original = link.get("url")
    if _norm_scheme(scheme) == "doi":
        guess = slug("DOI", publisher, ident) or slug(publisher) or "doi"
        candidates = datalink_doi_candidates(link)
        if not candidates:
            stats.dropped[("malformed DOI", guess)] += 1
            return None
        chosen = _choose_doi(candidates, handles)
        if chosen is PENDING:
            stats.doi_pending += 1
            return PENDING
        if chosen is None:
            stats.dropped[("DOI not registered", guess)] += 1
            return None
        link_id = chosen
        resource = slug("DOI", publisher, link_id) or slug(publisher) or "doi"
    else:
        resource = slug(scheme, publisher)
        if resource is None:
            stats.unmapped_scheme[scheme or publisher or "?"] += 1
            return None
        choice = _choose_other(ident, original, resource)
        if choice is None:
            stats.dropped[("malformed", resource)] += 1
            return None
        link_id, how = choice
        if how == "resolver":
            stats.from_resolver += 1
    if link_id != ident:
        stats.repaired[resource] += 1
    return {
        "resource": resource,
        "id": link_id,
        "url": canonical_url(resource, link_id, original),
        "title": link.get("title"),
        "obtained_by": link.get("obtained_by") or "ext_links",
        "relationship": link.get("relationship"),
        "section": None,
        "frequency": None,
        "_id_scheme": scheme,
        "_publisher": publisher,
        "_category": link.get("category"),
    }


def derived_biostudies(pmcid: str) -> dict | None:
    """Every PMC article with supplementary files has a BioStudies entry S-EPMC<digits>."""
    digits = (pmcid or "").strip().upper().removeprefix("PMC")
    if not digits.isdigit():
        return None
    accession = f"S-EPMC{digits}"
    return {
        "resource": "biostudies",
        "id": accession,
        "url": f"https://www.ebi.ac.uk/biostudies/studies/{accession}",
        "title": "Supplementary material",
        "obtained_by": "derived",
        "relationship": "IsSupplementedBy",
        "section": None,
        "frequency": None,
        "_id_scheme": "BioStudies",
        "_publisher": "BioStudies",
        "_category": None,
    }


# ---------------------------------------------------------------------------
# DOI confirmation at doi.org
# ---------------------------------------------------------------------------


def collect_doi_candidates(annotation_paths: tuple[Path, ...], datalinks_path: Path) -> set[str]:
    """Every DOI candidate the inputs could turn into a link, lowercased. Annotation lines that
    carry no DOI are skipped without being parsed."""
    wanted: set[str] = set()
    for path in annotation_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if ('"sub_type": "DOI"' not in line and '"sub_type": "doi"' not in line
                        and "identifiers.org/doi" not in line):
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for annotation in record.get("annotations") or []:
                    wanted.update(c.lower() for c in annotation_doi_candidates(annotation))
    if datalinks_path.exists():
        with datalinks_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for link in record.get("links") or []:
                    wanted.update(c.lower() for c in datalink_doi_candidates(link))
    return wanted


def load_handle_cache(path: Path, max_age_days: int) -> dict[str, bool]:
    """doi -> registered, for every verdict still trusted: a registration is permanent, a
    non-registration is re-asked once older than `max_age_days` (a DOI can be minted later)."""
    latest: dict[str, tuple[bool, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                latest[row["doi"]] = (row.get("exists") == "Y", row.get("checked_at") or "")
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    trusted: dict[str, bool] = {}
    for doi, (exists, checked_at) in latest.items():
        if exists:
            trusted[doi] = True
            continue
        try:
            if datetime.fromisoformat(checked_at) >= cutoff:
                trusted[doi] = False
        except ValueError:
            continue
    return trusted


def lookup_handle(session, doi: str) -> bool | None:
    """True registered, False not registered, None when doi.org gave no usable answer."""
    resp = session.get(HANDLE_API + quote(doi, safe="/"), timeout=HANDLE_TIMEOUT)
    if resp.status_code not in (200, 404):
        resp.raise_for_status()
        return None
    code = resp.json().get("responseCode")
    if code in (1, 200):
        return True
    if code == 100:
        return False
    return None


def verify_doi_handles(candidates: set[str], cache_path: Path, workers: int, max_age_days: int,
                       session=None) -> dict[str, bool]:
    """Confirms every candidate not already trusted in the cache, appending each verdict as it
    lands. A lookup that fails leaves its DOI without a verdict: the records linking to it are
    withheld this build and asked again on the next."""
    if not candidates:
        return {}
    trusted = load_handle_cache(cache_path, max_age_days)
    to_check = sorted(d for d in candidates if d not in trusted)
    print(f"build_data_links: {len(candidates):,} DOI candidates, {len(candidates) - len(to_check):,} "
          f"already known, {len(to_check):,} to confirm at doi.org ({workers} workers)", flush=True)
    if to_check:
        own_session = session is None
        session = session or _session(workers)
        checked_at = datetime.now(timezone.utc).isoformat()
        header_needed = not cache_path.exists()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        started, errors = time.time(), 0
        with cache_path.open("a", newline="", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            writer = csv.DictWriter(f, fieldnames=HANDLE_COLUMNS)
            if header_needed:
                writer.writeheader()
            futures = {pool.submit(lookup_handle, session, doi): doi for doi in to_check}
            for n, future in enumerate(as_completed(futures), 1):
                doi = futures[future]
                try:
                    verdict = future.result()
                except Exception:  # noqa: BLE001 -- a network failure is "no verdict", never "registered"
                    verdict = None
                if verdict is None:
                    errors += 1
                    continue
                trusted[doi] = verdict
                writer.writerow({"doi": doi, "exists": "Y" if verdict else "N", "checked_at": checked_at})
                if n % 500 == 0:
                    f.flush()
        if own_session:
            session.close()
        elapsed = time.time() - started
        print(f"build_data_links: {len(to_check):,} doi.org lookups in {elapsed:.1f}s "
              f"({len(to_check) / elapsed if elapsed else 0:.0f}/s), {errors} without a verdict"
              + (" -- records linking to those stay unresolved; re-run to retry" if errors else ""),
              flush=True)
    return {doi: trusted[doi] for doi in candidates if doi in trusted}


# ---------------------------------------------------------------------------
# Assembling one document's block
# ---------------------------------------------------------------------------


def assemble(links: list[dict], sources: list[str], fetched_at: str,
             browse_urls: dict[str, str] | None = None) -> dict:
    """Dedupe on (resource, id), summarise per resource, cap the detail. Deterministic: resources
    by descending count then slug, links in encounter order within a resource.

    The first route to find a link keeps it (`obtained_by`); a later route fills what the first
    lacked and adds itself to the resource's `routes`, so a card can say both found it. A source
    that counted more entries than it returned (`_unfetched`: an EBI Search entry list past its page
    cap) raises `count` and `link_count` to the true totals and marks the block truncated."""
    merged: dict[tuple[str, str], dict] = {}
    for link in links:
        if link is None:
            continue
        key = (link["resource"], link["id"].lower())
        routes = set(link.get("_routes") or ())
        if link.get("obtained_by"):
            routes.add(link["obtained_by"])
        if key in merged:
            kept = merged[key]
            for field in ("url", "title", "section", "frequency", "matched_by", "source_domain"):
                if not kept.get(field) and link.get(field):     # a later route may know more
                    kept[field] = link[field]
            kept["_routes"] |= routes
            kept["_unfetched"] = max(kept.get("_unfetched", 0), link.get("_unfetched", 0))
            continue
        merged[key] = dict(link, _routes=routes)

    by_resource: dict[str, list[dict]] = defaultdict(list)
    for link in merged.values():
        by_resource[link["resource"]].append(link)

    ordered = sorted(by_resource.items(), key=lambda item: (-len(item[1]), item[0]))
    resources, detail, truncated = [], [], False
    unfetched_total = 0
    for resource_slug, group in ordered:
        first = group[0]
        info = describe(resource_slug, first.get("_id_scheme") or first.get("_publisher"))
        unfetched = sum(link.get("_unfetched", 0) for link in group)
        unfetched_total += unfetched
        resources.append({
            "resource": resource_slug,
            "label": info.label,
            "category": info.category if info.category != "Other" and info.category
            else (first.get("_category") or info.category),
            "id_scheme": first.get("_id_scheme"),
            "publisher": first.get("_publisher"),
            "obtained_by": first.get("obtained_by"),
            "count": len(group) + unfetched,
            "routes": sorted(set().union(*(link["_routes"] for link in group))),
            "browse_url": (browse_urls or {}).get(resource_slug),
        })
        kept = group[:MAX_LINKS_PER_RESOURCE]
        truncated = truncated or len(kept) < len(group) or bool(unfetched)
        for link in kept:
            if len(detail) >= MAX_LINKS_PER_DOCUMENT:
                truncated = True
                break
            detail.append({k: link.get(k) for k in LINK_KEYS})

    return {
        "fetched_at": fetched_at,
        "sources": sources,
        "link_count": len(merged) + unfetched_total,
        "truncated": truncated,
        "resources": resources,
        "links": detail,
    }


def assert_clean(pid: str, detail: dict) -> None:
    """The build's own last check. A failure here is a bug in the normaliser, and raising before
    the output files are replaced keeps it out of every file a loader could read."""
    bad = malformed_links(detail)
    if bad:
        raise RuntimeError(
            f"build_data_links produced {len(bad)} malformed link(s) for {pid}, first {bad[0]} -- "
            f"the output files have not been replaced"
        )


# ---------------------------------------------------------------------------
# Loading the routes
# ---------------------------------------------------------------------------


_METADATA_COLUMNS = ("epmc_source", "epmc_id", "preprint_server", "pmcid", "has_data",
                     "data_links_tags", "accession_types", "db_cross_references",
                     "has_tm_accessions", "has_db_xrefs", "has_suppl", "fetched_at")
# The fetchers write `source`, then `id`, first on every line, so a line's identity can be read
# without parsing the rest of it.
_LINE_IDENTITY_RE = re.compile(r'^\{"source": "((?:[^"\\]|\\.)*)", "id": "((?:[^"\\]|\\.)*)"')


def load_metadata(path: Path, wanted: set[tuple[str, str]] | None = None) -> dict[tuple[str, str], dict]:
    """(key_type, key) -> the metadata columns, for the keys in `wanted` (all when None). A later
    fetch of the same key supersedes an earlier one."""
    index: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return index
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if not row.get("key_type"):
                break  # a staged batch CSV carries its metadata per row; nothing to index
            key = (row["key_type"], row["key"])
            if wanted is not None and key not in wanted:
                continue
            index[key] = {c: row.get(c) or "" for c in _METADATA_COLUMNS}
    return index


def line_identity(line: str) -> tuple[str, str] | None:
    match = _LINE_IDENTITY_RE.match(line)
    if match:
        return match.group(1), match.group(2)
    try:
        rec = json.loads(line)
        return rec["source"], rec["id"]
    except (ValueError, KeyError, TypeError):
        return None


def load_jsonl(path: Path, wanted: set[tuple[str, str]] | None = None) -> dict[tuple[str, str], dict]:
    """(source, id) -> the latest record, parsing only the lines whose identity is in `wanted`."""
    index: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            identity = line_identity(line)
            if identity is None or (wanted is not None and identity not in wanted):
                continue
            try:
                index[identity] = json.loads(line)
            except ValueError:
                continue
    return index


def shard_of(pid: str, shards: int) -> int:
    return zlib.crc32(pid.encode("utf-8")) % shards if shards > 1 else 0


def _flag(row: dict, column: str) -> bool:
    return (row.get(column) or "").strip().upper() == "Y"


def _metadata_for(row: dict, metadata: dict[tuple[str, str], dict]) -> dict | None:
    """A staged batch CSV carries its own metadata columns (Phase 4 of the pipeline); a corpus key
    file needs the metadata pass joined by key."""
    if (row.get("epmc_id") or "").strip() and "has_data" in row:
        return row
    key = metadata_key(row)
    return metadata.get(key) if key else None


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _process_row(row: dict, meta: dict | None, annotations: dict, datalinks: dict, bulk: dict,
                 datalinks_scope: str, stats: Stats, preprint_writer, data_writer,
                 handles: dict[str, bool], ebi: EbiRoute | None = None,
                 identifier_writer=None) -> None:
    pid = row["pid"].strip()
    stats.counts["documents"] += 1
    if meta is None:
        stats.counts["no_metadata"] += 1
        return
    source = (meta.get("epmc_source") or "").strip()
    ext_id = (meta.get("epmc_id") or "").strip()
    # Only a PPR record's publisher is a preprint server (see fetch_epmc_metadata.identity_fields);
    # enforced here too so a metadata file fetched before that rule cannot leak one through.
    server = (meta.get("preprint_server") or "").strip() if source == "PPR" else ""
    if source and ext_id:
        stats.counts["identified"] += 1
        if preprint_writer is not None:
            preprint_writer.writerow({
                "pid": pid, "epmc_source": source, "epmc_id": ext_id, "preprint_server": server})
        if server:
            stats.counts["with_preprint_server"] += 1
    else:
        stats.counts["epmc_miss"] += 1

    out = {
        "pid": pid,
        "has_data": (meta.get("has_data") or "").strip().upper(),
        "data_links_tags": meta.get("data_links_tags") or "[]",
        "accession_types": meta.get("accession_types") or "[]",
        "db_cross_references": meta.get("db_cross_references") or "[]",
        "data_links_json": "",
    }

    identity = (source, ext_id) if source and ext_id else None
    targeted_ann = identity is not None and _flag(meta, "has_tm_accessions")
    targeted_dl = (identity is not None and datalinks_scope != "none"
                   and datalinks_targeted(meta, datalinks_scope))
    ann = annotations.get(identity) if identity else None
    dl = datalinks.get(identity) if identity else None
    bulk_rec = bulk.get(identity) if identity else None

    # EBI Search (v1.5.0), keyed on OUR identifiers from the keys row. A record in scope whose EBI
    # Search route has not fully answered is withheld like any other half-fetched record.
    covered = ebi is not None and ebi.covers(pid)
    ebi_links: list[dict] = []
    ebi_answered, ebi_times = True, []
    if covered:
        stats.counts["ebisearch_covered"] += 1
        ebi_links, ebi_answered, ebi_times = ebi.links_for(row.get("pmid"), row.get("pmcid"),
                                                           row.get("doi"), stats)
        if not ebi_answered:
            stats.counts["unresolved_ebisearch"] += 1

    routes_answered = ((not targeted_ann or ann is not None)
                       and (not targeted_dl or dl is not None) and ebi_answered)
    if not routes_answered:
        stats.counts["unresolved"] += 1
    else:
        links: list = []
        sources: list[str] = []
        timestamps = [meta.get("fetched_at") or ""]
        if ann is not None:
            sources.append(SOURCE_ANNOTATIONS)
            timestamps.append(ann.get("fetched_at") or "")
            links += [link_from_annotation(a, stats, handles) for a in ann.get("annotations") or []]
        if dl is not None:
            sources.append(SOURCE_DATALINKS)
            timestamps.append(dl.get("fetched_at") or "")
            links += [link_from_datalink(d, stats, handles) for d in dl.get("links") or []]
        if bulk_rec is not None:
            sources.append(SOURCE_BULK)
            timestamps.append(bulk_rec.get("fetched_at") or "")
            links += [link_from_annotation(a, stats, handles) for a in bulk_rec.get("annotations") or []]
        if identity and _flag(meta, "has_suppl"):
            derived = derived_biostudies(meta.get("pmcid") or row.get("pmcid") or "")
            if derived is not None:
                sources.append(SOURCE_DERIVED)
                links.append(derived)
        if any(link is PENDING for link in links):
            stats.counts["unresolved"] += 1
            stats.counts["unresolved_pending_doi"] += 1
        else:
            if not sources:
                sources = [SOURCE_SEARCH]
            if covered:
                sources.append(SOURCE_EBISEARCH)
                timestamps += ebi_times
                links += ebi_links
            # Every route's links filed under their home resource, so the dedupe sees one accession.
            links = resolve_pxd_hosts([canonicalise_link(link) if isinstance(link, dict) else link
                                       for link in links])
            if covered:
                _count_ebisearch(links, stats)
            browse: dict[str, str] = {}
            for resource_slug in {link["resource"] for link in links if isinstance(link, dict)}:
                url = browse_url(resource_slug, row.get("pmid"))
                if url:
                    browse[resource_slug] = url
            dated = [t for t in timestamps if t]
            detail = assemble(links, sources, max(dated) if dated else "", browse)
            assert_clean(pid, detail)
            out["data_links_json"] = json.dumps(detail, ensure_ascii=False)
            if covered:
                # Written with the links, never for a withheld record: both files say the same.
                _write_identifier(identifier_writer, pid, row, ebi_links, stats)
            stats.counts["resolved"] += 1
            if detail["resources"]:
                stats.counts["with_resources"] += 1
            if detail["truncated"]:
                stats.counts["truncated"] += 1
            for resource in detail["resources"]:
                stats.resources[resource["resource"]] += 1
    if data_writer is not None:
        data_writer.writerow(out)


def _count_ebisearch(links: list, stats: Stats) -> None:
    """Per resource, the documents where EBI Search added a link no Europe PMC route had, and those
    where both found the same one. `links` are canonicalised and not yet deduplicated; only EBI
    Search links carry a `source_domain`."""
    def key(link: dict) -> tuple[str, str]:
        return link["resource"], str(link["id"]).lower()
    candidates = [link for link in links if isinstance(link, dict)]
    held = {key(link) for link in candidates if not link.get("source_domain")}
    ebi = [link for link in candidates if link.get("source_domain")]
    if not ebi:
        return
    stats.counts["ebisearch_with_links"] += 1
    gained = {link["resource"] for link in ebi if key(link) not in held}
    if gained:
        stats.counts["ebisearch_gaining"] += 1
    stats.ebisearch_resources.update(gained)
    stats.ebisearch_confirmed.update({link["resource"] for link in ebi if key(link) in held})


def _write_identifier(writer, pid: str, row: dict, ebi_links: list[dict], stats: Stats) -> None:
    """`identifiers.dome_registry` for one in-scope record: the DOME Registry entry naming the paper
    (the first in sort order when several do; the card keeps them all), `""` when the paper could be
    looked up -- a PMID or PMCID, the only keys the registry's EBI Search entries carry -- and none
    names it, and no row for a paper that cannot be looked up at all. Counted without a writer too,
    so `--report-only` reports what a build would write."""
    ids = sorted({link["id"] for link in ebi_links if link["resource"] == "dome_registry"})
    if ids:
        value = ids[0]
        stats.counts["dome_registry_ids"] += 1
    elif paper_keys(row.get("pmid"), row.get("pmcid"), None):
        value = ""
    else:
        return
    if writer is not None:
        writer.writerow({"pid": pid, "dome_registry": value})
    stats.counts["identifiers_written"] += 1


def _row_carries_metadata(row: dict) -> bool:
    return bool((row.get("epmc_id") or "").strip()) and "has_data" in row


def in_scope_pids(keys_path: Path, scope: str, classification_events: Path | None) -> set[str] | None:
    """The pids the EBI Search route covers: every row for `all`; for `positives`, the rows whose
    verdict is positive -- the keys file's `classification` column, or a staged batch's event log
    when one is given. A scope that cannot be decided stops the build rather than silently covering
    nothing or everything."""
    if scope == "all":
        return None
    verdicts = latest_verdicts(classification_events) if classification_events else None
    positives: set[str] = set()
    with keys_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if verdicts is None and "classification" not in (reader.fieldnames or []):
            raise SystemExit(
                f"build_data_links: --ebisearch-scope positives needs a classification, and "
                f"{keys_path.name} has no classification column -- pass --classification-events "
                f"<the batch's events CSV>, or --ebisearch-scope none")
        for row in reader:
            pid = (row.get("pid") or "").strip()
            verdict = (verdicts.get(pid) if verdicts is not None
                       else (row.get("classification") or "").strip())
            if pid and verdict == "positive":
                positives.add(pid)
    return positives


def build(keys_path: Path, metadata_path: Path, annotations_path: Path, datalinks_path: Path,
          bulk_path: Path, datalinks_scope: str, report_only: bool,
          out_preprints: Path, out_data_links: Path, shards: int = 1, *,
          handles_path: Path = DEFAULT_HANDLES, handle_workers: int = DEFAULT_HANDLE_WORKERS,
          handle_max_age_days: int = DEFAULT_HANDLE_MAX_AGE_DAYS, session=None,
          ebisearch_scope: str = "none",
          ebisearch_discovery: Path = DEFAULT_EBISEARCH_DISCOVERY,
          ebisearch_detail: Path = DEFAULT_EBISEARCH_DETAIL,
          ebisearch_domains_dir: Path = DEFAULT_EBISEARCH_DOMAINS,
          classification_events: Path | None = None,
          out_identifiers: Path = DEFAULT_OUT_IDENTIFIERS,
          require_all_dumps: bool = True) -> Stats:
    """`ebisearch_scope` defaults to "none" for a library caller; the CLI defaults to "positives"."""
    stats = Stats()
    ebi = None
    if ebisearch_scope != "none":
        in_scope = in_scope_pids(keys_path, ebisearch_scope, classification_events)
        ebi = EbiRoute.load(keys_path, in_scope, ebisearch_discovery, ebisearch_detail,
                            ebisearch_domains_dir, require_all_dumps=require_all_dumps)
        stats.ebisearch_rejected = ebi.rejected
        stats.ebisearch_unclassified = ebi.dumps.unclassified
        covered = "every record" if in_scope is None else f"{len(in_scope):,} records"
        print(f"build_data_links: EBI Search route ({ebisearch_scope}): {covered} in scope, "
              f"{len(ebi.discovery):,} discovery / {len(ebi.detail):,} detail records, "
              f"{len(ebi.dumps.fetched_at)} dumped domains", flush=True)
    candidates = collect_doi_candidates((annotations_path, bulk_path), datalinks_path)
    handles = verify_doi_handles(candidates, handles_path, handle_workers, handle_max_age_days,
                                 session)

    sinks: list = []
    preprint_writer = data_writer = identifier_writer = None
    tmp_p = out_preprints.with_suffix(out_preprints.suffix + ".tmp")
    tmp_d = out_data_links.with_suffix(out_data_links.suffix + ".tmp")
    tmp_i = out_identifiers.with_suffix(out_identifiers.suffix + ".tmp")
    if not report_only:
        out_preprints.parent.mkdir(parents=True, exist_ok=True)
        preprint_sink = tmp_p.open("w", newline="", encoding="utf-8")
        data_sink = tmp_d.open("w", newline="", encoding="utf-8")
        sinks += [preprint_sink, data_sink]
        preprint_writer = csv.DictWriter(preprint_sink, fieldnames=PREPRINT_COLUMNS)
        data_writer = csv.DictWriter(data_sink, fieldnames=DATA_LINKS_COLUMNS)
        preprint_writer.writeheader()
        data_writer.writeheader()
        if ebi is not None:
            out_identifiers.parent.mkdir(parents=True, exist_ok=True)
            identifier_sink = tmp_i.open("w", newline="", encoding="utf-8")
            sinks.append(identifier_sink)
            identifier_writer = csv.DictWriter(identifier_sink, fieldnames=IDENTIFIER_COLUMNS)
            identifier_writer.writeheader()

    try:
        for shard in range(shards):
            rows: list[dict] = []
            seen: set[str] = set()
            with keys_path.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    pid = (row.get("pid") or "").strip()
                    if not pid or shard_of(pid, shards) != shard:
                        continue
                    if pid in seen:            # a duplicate pid always hashes to the same shard
                        stats.counts["skipped_no_pid_or_duplicate"] += 1
                        continue
                    seen.add(pid)
                    rows.append(row)

            wanted_meta = {k for k in (metadata_key(r) for r in rows if not _row_carries_metadata(r))
                           if k is not None}
            metadata = load_metadata(metadata_path, wanted_meta) if wanted_meta else {}
            metas: list[dict | None] = []
            identities: set[tuple[str, str]] = set()
            for row in rows:
                if _row_carries_metadata(row):
                    meta = row
                else:
                    key = metadata_key(row)
                    meta = metadata.get(key) if key else None
                metas.append(meta)
                if meta and (meta.get("epmc_source") or "").strip() and (meta.get("epmc_id") or "").strip():
                    identities.add((meta["epmc_source"].strip(), meta["epmc_id"].strip()))

            annotations = load_jsonl(annotations_path, identities)
            datalinks = load_jsonl(datalinks_path, identities)
            bulk = load_jsonl(bulk_path, identities)
            for row, meta in zip(rows, metas):
                _process_row(row, meta, annotations, datalinks, bulk, datalinks_scope, stats,
                             preprint_writer, data_writer, handles, ebi, identifier_writer)
            print(f"build_data_links: shard {shard + 1}/{shards}: {len(rows):,} documents, "
                  f"{len(metadata):,} metadata rows, {len(annotations):,} annotation / "
                  f"{len(datalinks):,} datalinks / {len(bulk):,} bulk records", flush=True)
            del rows, metas, metadata, annotations, datalinks, bulk
    finally:
        for sink in sinks:
            sink.close()

    if not report_only:
        os.replace(tmp_p, out_preprints)
        os.replace(tmp_d, out_data_links)
        if ebi is not None:
            os.replace(tmp_i, out_identifiers)
    return stats


def render(stats: Stats, report_only: bool, out_preprints: Path, out_data_links: Path,
           out_identifiers: Path | None = None) -> None:
    c = stats.counts
    print(f"\nbuild_data_links: {c['documents']:,} documents")
    print(f"  no metadata yet        : {c['no_metadata']:,}  (run fetch_epmc_metadata.py)")
    print(f"  Europe PMC identity    : {c['identified']:,}  ({c['epmc_miss']:,} looked up, no record)")
    print(f"  with a preprint server : {c['with_preprint_server']:,}")
    print(f"  data links resolved    : {c['resolved']:,}  ({c['with_resources']:,} with resources, "
          f"{c['truncated']:,} capped)")
    print(f"  unresolved (a targeted route or a DOI check has not answered yet): {c['unresolved']:,}"
          + (f"  ({c['unresolved_pending_doi']:,} waiting on doi.org)"
             if c["unresolved_pending_doi"] else ""))
    print(f"  literature DOIs skipped: {stats.reference_dois:,}")
    print(f"  identifiers repaired   : {sum(stats.repaired.values()):,}  "
          f"({stats.from_resolver:,} taken from Europe PMC's resolver URL)")
    if stats.dropped:
        print("\n  links dropped (nothing clean recoverable, or DOI not registered at doi.org):")
        for (reason, resource), n in stats.dropped.most_common(25):
            print(f"    {reason:<24} {resource:<22} {n:>8,}")
    if stats.repaired:
        print("\n  repaired identifiers by resource:", dict(stats.repaired.most_common(15)))
    if stats.resources:
        print("\n  resources (documents carrying each):")
        for name, n in stats.resources.most_common(40):
            print(f"    {name:<22} {n:>9,}   {describe(name).label} / {describe(name).category}")
    if c["ebisearch_covered"]:
        print(f"\n  EBI Search route       : {c['ebisearch_covered']:,} records in scope, "
              f"{c['ebisearch_with_links']:,} with its links, {c['ebisearch_gaining']:,} gaining a link "
              f"no Europe PMC route had; {c['unresolved_ebisearch']:,} withheld "
              f"({c['ebisearch_waiting_discovery']:,} not discovered, {c['ebisearch_waiting_detail']:,} "
              f"domain pairs without complete detail); {c['ebisearch_failed_discovery']:,} failed "
              f"discovery records")
        print(f"  identifiers.dome_registry: {c['dome_registry_ids']:,} entries, "
              f"{c['identifiers_written'] - c['dome_registry_ids']:,} looked up with none")
        if stats.ebisearch_resources:
            print("\n  links EBI Search added (documents gaining a link to each resource):")
            for name, n in stats.ebisearch_resources.most_common(40):
                print(f"    {name:<22} {n:>9,}   {describe(name).label} / {describe(name).category}")
        if stats.ebisearch_confirmed:
            print("\n  links Europe PMC and EBI Search both found (documents, per resource):")
            for name, n in stats.ebisearch_confirmed.most_common(20):
                print(f"    {name:<22} {n:>9,}   {describe(name).label} / {describe(name).category}")
        if stats.ebisearch_rejected:
            print("\n  EBI Search domains rejected (not in ebisearch_resources.DOMAINS; papers listing each):")
            for name, n in stats.ebisearch_rejected.most_common(30):
                print(f"    {name:<34} {n:>9,}")
        if stats.ebisearch_unclassified:
            print("\n  dump publication values of no known shape (ignored):")
            for (domain, field), n in stats.ebisearch_unclassified.most_common(15):
                print(f"    {domain:<30} {field:<12} {n:>7,}")
    if stats.unmapped_scheme:
        print("\n  UNMAPPED schemes / publishers (add to datalinks_resources.ALIASES):")
        for name, n in stats.unmapped_scheme.most_common(30):
            print(f"    {name!r:<30} {n:>9,}")
    if stats.unmapped_doi_prefix:
        print("\n  DOI prefixes not in DOI_PREFIXES (literature, or a repository to add):")
        for name, n in stats.unmapped_doi_prefix.most_common(30):
            print(f"    {name:<14} {n:>9,}")
    if report_only:
        print("\nbuild_data_links: --report-only, nothing written.")
    else:
        written = [str(out_preprints), str(out_data_links)]
        if c["ebisearch_covered"] and out_identifiers is not None:
            written.append(str(out_identifiers))
        print(f"\nbuild_data_links: wrote {', '.join(written)}")
        print("next: python3 load_fields.py --mode preprints   /   --mode data_links"
              + ("   /   --mode identifiers" if c["ebisearch_covered"] else "") + "   (dry run first)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS,
                        help="corpus_keys.csv, or a staged incoming CSV (pid, pmid, pmcid, doi, ...)")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--datalinks", type=Path, default=DEFAULT_DATALINKS)
    parser.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--datalinks-scope", choices=("residual", "supporting", "all", "none"),
                        default="residual",
                        help="Which records fetch_datalinks.py was run for; a targeted record with "
                             "no answer yet is left unresolved rather than half-written. 'none' "
                             "builds without the /datalinks route (when that endpoint is down); a "
                             "later build with it re-dates and completes the records.")
    parser.add_argument("--out-preprints", type=Path, default=DEFAULT_OUT_PREPRINTS)
    parser.add_argument("--out-data-links", type=Path, default=DEFAULT_OUT_DATA_LINKS)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--shards", type=int, default=8,
                        help="Split documents into N shards by pid hash so only one shard's inputs "
                             "are held in memory at a time. 1 for a small batch.")
    parser.add_argument("--handles", type=Path, default=DEFAULT_HANDLES,
                        help="The doi.org verdict cache (doi, exists, checked_at).")
    parser.add_argument("--handle-workers", type=int, default=DEFAULT_HANDLE_WORKERS)
    parser.add_argument("--handle-max-age-days", type=int, default=DEFAULT_HANDLE_MAX_AGE_DAYS,
                        help="Re-ask doi.org about a DOI it said was not registered once the verdict "
                             "is this old. A registration is never re-asked.")
    parser.add_argument("--ebisearch-scope", choices=("positives", "all", "none"), default="positives",
                        help="Which records get EBI Search's database-side links (schema v1.5.0). "
                             "'positives' reads the keys file's classification column, or "
                             "--classification-events for a staged batch; 'none' builds without "
                             "the route.")
    parser.add_argument("--ebisearch-discovery", type=Path, default=DEFAULT_EBISEARCH_DISCOVERY)
    parser.add_argument("--ebisearch-detail", type=Path, default=DEFAULT_EBISEARCH_DETAIL)
    parser.add_argument("--ebisearch-domains", type=Path, default=DEFAULT_EBISEARCH_DOMAINS)
    parser.add_argument("--classification-events", type=Path, default=None,
                        help="A staged batch's classification event log: the verdicts that decide "
                             "the EBI Search scope when the keys file carries none.")
    parser.add_argument("--out-identifiers", type=Path, default=DEFAULT_OUT_IDENTIFIERS)
    args = parser.parse_args()
    stats = build(args.keys, args.metadata, args.annotations, args.datalinks, args.bulk,
                  args.datalinks_scope, args.report_only, args.out_preprints, args.out_data_links,
                  max(1, args.shards), handles_path=args.handles,
                  handle_workers=args.handle_workers, handle_max_age_days=args.handle_max_age_days,
                  ebisearch_scope=args.ebisearch_scope,
                  ebisearch_discovery=args.ebisearch_discovery,
                  ebisearch_detail=args.ebisearch_detail,
                  ebisearch_domains_dir=args.ebisearch_domains,
                  classification_events=args.classification_events,
                  out_identifiers=args.out_identifiers)
    render(stats, args.report_only, args.out_preprints, args.out_data_links, args.out_identifiers)


if __name__ == "__main__":
    main()
