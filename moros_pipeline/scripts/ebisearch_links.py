"""The EBI Search route of the data-links merge: reads what `fetch_ebisearch_xrefs.py` and
`fetch_ebisearch_domains.py` wrote and turns it into links in the shape
`build_data_links.assemble()` takes, for the accepted domains in `ebisearch_resources.py` only.

Two routes, both keyed on OUR identifiers for the paper, never on EBI Search's echo:

- **xref**: `europepmc/entry/{pmid}/xref` discovery says which domains name the PMID, and `detail`
  holds their entries. Used for the accepted domains too large to dump (`XREF_DOMAINS`). A record
  is answered only once its discovery record exists and every accepted domain it lists has a
  complete detail record, so a batch built between `discover` and `detail` is withheld rather
  than stamped complete. A discovery record marked `failed` (a PMID EBI Search errors on every
  time) answers with no xref links.
- **domain**: whole dumps of the smaller accepted domains, indexed once on the publication values
  their entries carry. Values are classified by SHAPE, not by field name, because the fields are
  not what they say (measured 2026-09-14): EMDB stores `doi:`-prefixed DOIs, NODE puts DOIs in
  PUBMED, ArrayExpress puts PMIDs in DOI, EVA writes `PubMed:41603203`, BioStudies stores DOI
  URLs. No accepted domain carries a PPR field, so a preprint is reachable only by its DOI.

Pure over its inputs: no network.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

from ebisearch_resources import (
    DOMAINS,
    DUMP_DOMAINS,
    OBTAINED_BY_DOMAIN,
    OBTAINED_BY_XREF,
    XREF_DOMAINS,
    canonicalise,
    entry_url,
    relationship_for,
)
from link_identifiers import DOI_RE, problems, url_problems

csv.field_size_limit(sys.maxsize)

PUBLICATION_FIELDS = frozenset({
    "PUBMED", "PMID", "PUB_MED", "PUBMED_ID", "MEDLINE", "PIMD", "EUROPE_PMC", "EUROPEPMC",
    "PMC", "PMCID", "PPR", "DOI",
})
KEY_ORDER = ("pmid", "pmcid", "doi")      # the preference when one entry names a paper twice

_PUBMED_PREFIX_RE = re.compile(r"^(?:pubmed|pmid)\s*:\s*", re.I)
_DOI_PREFIX_RE = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/|(?:dx\.)?doi\.org/)", re.I)
_PMC_RE = re.compile(r"^PMC\d+$", re.I)
_PUBMED_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:ncbi\.nlm\.nih\.gov/pubmed|pubmed\.ncbi\.nlm\.nih\.gov)/(\d+)$", re.I)
_SPACE_AFTER_SLASH_RE = re.compile(r"(?<=/)\s+")
# The fetchers write `source`, then `id`, first on every line.
_LINE_ID_RE = re.compile(r'^\{"source": "[^"]*", "id": "([^"]*)"')


def classify_value(value) -> tuple[str, str] | None:
    """A publication value as (kind, normalised value): ("pmid", "38427602"), ("pmcid",
    "PMC3232365"), ("doi", "10.1073/pnas.2320493121"); None for "n/a", "in preparation",
    "2000-02-29" and anything else.

    Repositories wrap ids in ways measured in the dumps (2026-09-14): PubMed URLs
    (ArrayExpress), stacked prefixes such as `doi:https://doi.org/10...` (EMDB), form-encoded
    values wrapped in `+` (BioModels, BioStudies), and a space after the DOI's slash (NODE). A
    truncated or concatenated id is not guessed at."""
    text = unquote(str(value or "")).strip().strip("+").strip().rstrip("/").strip()
    if not text:
        return None
    text = _PUBMED_PREFIX_RE.sub("", text)
    pubmed_url = _PUBMED_URL_RE.match(text)
    if pubmed_url:
        text = pubmed_url.group(1)
    if text.isdigit():
        return "pmid", text
    if _PMC_RE.match(text):
        return "pmcid", text.upper()
    doi = text
    while True:
        stripped = _DOI_PREFIX_RE.sub("", doi, count=1).strip()
        if stripped == doi:
            break
        doi = stripped
    doi = _SPACE_AFTER_SLASH_RE.sub("", doi)
    if DOI_RE.match(doi):
        return "doi", doi.lower()
    return None


def paper_keys(pmid, pmcid, doi) -> list[tuple[str, str]]:
    """Our keys for a paper, normalised exactly as `classify_value` normalises an entry's."""
    keys: list[tuple[str, str]] = []
    p = (pmid or "").strip()
    if p.isdigit():
        keys.append(("pmid", p))
    c = (pmcid or "").strip().upper()
    if c.isdigit():
        c = "PMC" + c
    if _PMC_RE.match(c):
        keys.append(("pmcid", c))
    d = (doi or "").strip().lower()
    if DOI_RE.match(d):
        keys.append(("doi", d))
    return keys


def _first(values) -> str | None:
    return next((str(v).strip() for v in values or [] if str(v or "").strip()), None)


def _records_for(path: Path, wanted: set[str]) -> Iterator[dict]:
    """Records whose `id` is in `wanted`, parsing only those lines."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            match = _LINE_ID_RE.match(line)
            if match and match.group(1) not in wanted:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if str(record.get("id")) in wanted:
                yield record


def load_discovery(path: Path, pmids: set[str]) -> tuple[dict[str, dict], Counter]:
    """pmid -> {fetched_at, failed, counts: {accepted xref domain: entries}}, latest line winning;
    and the rejected domains the records list (papers per domain), for the report."""
    latest: dict[str, dict] = {}
    for record in _records_for(path, pmids):
        latest[str(record["id"])] = record
    out: dict[str, dict] = {}
    rejected: Counter = Counter()
    for pmid, record in latest.items():
        counts: dict[str, int] = {}
        for domain in record.get("domains") or []:
            name, n = domain.get("id"), int(domain.get("referenceEntryCount") or 0)
            if not name or n <= 0:
                continue
            if name in XREF_DOMAINS:
                counts[name] = n
            elif name not in DOMAINS:
                rejected[name] += 1
        out[pmid] = {"fetched_at": record.get("fetched_at") or "",
                     "failed": bool(record.get("failed")), "counts": counts}
    return out, rejected


def load_detail(path: Path, pmids: set[str]) -> dict[tuple[str, str], dict]:
    """(pmid, accepted xref domain) -> the latest detail record."""
    latest: dict[tuple[str, str], dict] = {}
    for record in _records_for(path, pmids):
        if record.get("domain") in XREF_DOMAINS:
            latest[(str(record["id"]), record["domain"])] = record
    return latest


class DumpIndex:
    """(kind, value) -> the accepted dump entries naming it, as (domain, entry id, label,
    full_dataset_link). Built once over the keys asked for, so memory scales with the papers in
    scope, not with the dumps."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], list[tuple[str, str, str | None, str | None]]] = \
            defaultdict(list)
        self.fetched_at: dict[str, str] = {}
        self.unclassified: Counter = Counter()

    @classmethod
    def load(cls, dumps_dir: Path, wanted: set[tuple[str, str]] | None = None,
             domains=DUMP_DOMAINS) -> "DumpIndex":
        index = cls()
        manifest_path = dumps_dir / "_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        for domain in sorted(domains):
            path = dumps_dir / f"{domain}.jsonl"
            if not path.exists():
                continue
            index.fetched_at[domain] = (manifest.get(domain) or {}).get("fetched_at") or ""
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    fields = entry.get("fields") or {}
                    label = _first(fields.get("name")) or _first(fields.get("title"))
                    dataset_link = _first(fields.get("full_dataset_link"))
                    seen: set[tuple[str, str]] = set()
                    for field, values in fields.items():
                        if field.upper() not in PUBLICATION_FIELDS:
                            continue
                        for value in values or []:
                            key = classify_value(value)
                            if key is None:
                                if str(value or "").strip():
                                    index.unclassified[(domain, field)] += 1
                                continue
                            if key in seen or (wanted is not None and key not in wanted):
                                continue
                            seen.add(key)
                            index.entries[key].append(
                                (domain, str(entry.get("id")), label, dataset_link))
        return index


def make_link(domain: str, entry_id, title: str | None, dataset_link: str | None,
              obtained_by: str, matched_by: str, stats) -> dict | None:
    """One accepted entry -> a link, or None (counted) when its id is not clean."""
    info = DOMAINS[domain]
    raw = str(entry_id or "").strip()
    slug, link_id = canonicalise(info.slug, raw)
    if problems(link_id, slug):
        stats.dropped[("malformed (EBI Search)", slug)] += 1
        return None
    url = None
    if (slug, link_id) == (info.slug, raw) and dataset_link and not url_problems(dataset_link):
        url = dataset_link              # the repository's own link for this entry
    return {
        "resource": slug,
        "id": link_id,
        "url": url or entry_url(slug, link_id),
        "title": title or None,
        "obtained_by": obtained_by,
        "relationship": relationship_for(info),
        "section": None,
        "frequency": None,
        "matched_by": matched_by,
        "source_domain": domain,
        "_id_scheme": info.id_scheme,
        "_publisher": info.publisher,
        "_category": None,
        "_routes": {obtained_by},
    }


def dedupe_within_route(links: list[dict]) -> list[dict]:
    """One link per (resource, id) from EBI Search: a bio.tools entry naming the paper by PMID,
    PMCID and DOI is one link, kept with the strongest key."""
    best: dict[tuple[str, str], dict] = {}
    for link in links:
        key = (link["resource"], link["id"].lower())
        kept = best.get(key)
        if kept is None:
            best[key] = link
            continue
        routes = kept["_routes"] | link["_routes"]
        unfetched = max(kept.get("_unfetched", 0), link.get("_unfetched", 0))
        if KEY_ORDER.index(link["matched_by"]) < KEY_ORDER.index(kept["matched_by"]):
            kept = best[key] = dict(link)
        kept["_routes"] = routes
        if unfetched:
            kept["_unfetched"] = unfetched
    return list(best.values())


class EbiRoute:
    """The EBI Search inputs a build needs, loaded once, and the links for one record."""

    def __init__(self, in_scope: set[str] | None, discovery: dict[str, dict],
                 detail: dict[tuple[str, str], dict], dumps: DumpIndex, rejected: Counter) -> None:
        self.in_scope = in_scope
        self.discovery = discovery
        self.detail = detail
        self.dumps = dumps
        self.rejected = rejected

    @classmethod
    def load(cls, keys_path: Path, in_scope: set[str] | None, discovery_path: Path,
             detail_path: Path, dumps_dir: Path, require_all_dumps: bool = True) -> "EbiRoute":
        missing = sorted(d for d in DUMP_DOMAINS if not (dumps_dir / f"{d}.jsonl").exists())
        if missing and require_all_dumps:
            raise SystemExit(
                f"ebisearch_links: {len(missing)} accepted domain(s) not dumped in {dumps_dir}: "
                f"{', '.join(missing)} -- run fetch_ebisearch_domains.py first")
        if not discovery_path.exists():
            raise SystemExit(f"ebisearch_links: {discovery_path} not found -- run "
                             f"fetch_ebisearch_xrefs.py discover first")
        pmids: set[str] = set()
        keys: set[tuple[str, str]] = set()
        with keys_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid = (row.get("pid") or "").strip()
                if not pid or (in_scope is not None and pid not in in_scope):
                    continue
                for kind, value in paper_keys(row.get("pmid"), row.get("pmcid"), row.get("doi")):
                    keys.add((kind, value))
                    if kind == "pmid":
                        pmids.add(value)
        discovery, rejected = load_discovery(discovery_path, pmids)
        detail = load_detail(detail_path, pmids)
        dumps = DumpIndex.load(dumps_dir, keys)
        return cls(in_scope, discovery, detail, dumps, rejected)

    def covers(self, pid: str) -> bool:
        return self.in_scope is None or pid in self.in_scope

    def links_for(self, pmid, pmcid, doi, stats) -> tuple[list[dict], bool, list[str]]:
        """(links, answered, timestamps) for one in-scope record."""
        links: list[dict] = []
        timestamps: list[str] = []
        answered = True
        pmid = (pmid or "").strip()
        if pmid.isdigit():
            record = self.discovery.get(pmid)
            if record is None:
                answered = False
                stats.counts["ebisearch_waiting_discovery"] += 1
            else:
                timestamps.append(record["fetched_at"])
                if record["failed"]:
                    stats.counts["ebisearch_failed_discovery"] += 1
                for domain in sorted(record["counts"]):
                    detail = self.detail.get((pmid, domain))
                    if detail is None or not detail.get("complete"):
                        answered = False
                        stats.counts["ebisearch_waiting_detail"] += 1
                        continue
                    timestamps.append(detail.get("fetched_at") or "")
                    references = detail.get("references") or []
                    kept = []
                    for ref in references:
                        link = make_link(domain, ref.get("acc") or ref.get("id"),
                                         _first((ref.get("fields") or {}).get("name")), None,
                                         OBTAINED_BY_XREF, "pmid", stats)
                        if link is not None:
                            kept.append(link)
                    unfetched = max(0, int(detail.get("reference_count") or 0) - len(references))
                    if kept and unfetched:
                        kept[0]["_unfetched"] = unfetched
                    links += kept
        matched_domains: set[str] = set()
        for kind, value in paper_keys(pmid, pmcid, doi):
            for domain, entry_id, title, dataset_link in self.dumps.entries.get((kind, value), ()):
                link = make_link(domain, entry_id, title, dataset_link, OBTAINED_BY_DOMAIN, kind,
                                 stats)
                if link is not None:
                    links.append(link)
                    matched_domains.add(domain)
        timestamps += [self.dumps.fetched_at.get(d) or "" for d in matched_domains]
        return dedupe_within_route(links), answered, [t for t in timestamps if t]
