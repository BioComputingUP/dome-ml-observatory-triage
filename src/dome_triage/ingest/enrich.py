"""Fills in title/abstract/journal/year/authors for records that only carry IDs (the id_pair_only
and pdf_directory_gold adapters, and any dome_registry_api_json rows missing a PMCID) via a
batched Europe PMC lookup, with an NCBI ID Converter fallback to derive a PMCID from a DOI first.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from dome_triage.ingest.epmc_client import EpmcClient
from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid, clean_pmid, doi_to_pmcid
from dome_triage.ontology.mesh import extract_mesh_headings
from dome_triage.schema import RawRecord


def _needs_enrichment(record: RawRecord) -> bool:
    return not record.title or not record.abstract


def enrich_missing_metadata(records: list[RawRecord], client: EpmcClient) -> list[RawRecord]:
    to_enrich = [r for r in records if _needs_enrichment(r)]
    if not to_enrich:
        return records

    for record in to_enrich:
        if not record.pmcid and record.doi:
            record.pmcid = doi_to_pmcid(record.doi)

    by_pmcid = {r.pmcid: r for r in to_enrich if r.pmcid}
    by_pmid = {r.pmid: r for r in to_enrich if not r.pmcid and r.pmid}

    if by_pmcid:
        found = client.get_by_ids(list(by_pmcid.keys()), id_type="pmcid")
        for pmcid, result in found.items():
            _apply_epmc_result(by_pmcid[pmcid], result)

    if by_pmid:
        found = client.get_by_ids(list(by_pmid.keys()), id_type="pmid")
        for pmid, result in found.items():
            _apply_epmc_result(by_pmid[pmid], result)

    return records


def _apply_epmc_result(record: RawRecord, result: dict) -> None:
    record.title = record.title or result.get("title")
    record.abstract = record.abstract or result.get("abstractText")
    journal_info = result.get("journalInfo") or {}
    record.journal = record.journal or (journal_info.get("journal") or {}).get("title")
    record.authors = record.authors or result.get("authorString")
    record.doi = record.doi or result.get("doi")
    record.pmcid = record.pmcid or clean_pmcid(result.get("pmcid"))
    if not record.year and result.get("pubYear"):
        try:
            record.year = int(result["pubYear"])
        except (TypeError, ValueError):
            pass


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    return isinstance(value, str) and not value.strip()


def _non_null_strings(values: list | None) -> list[str]:
    return [v for v in (values or []) if isinstance(v, str)]


def enrich_missing_canonical_metadata(
    df: pd.DataFrame, client: EpmcClient, show_progress: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Fills in missing title/abstract/journal/authors/year/mesh_headings/pub_types/
    keywords_author/is_open_access directly on `canonical_dataset.csv` rows that have at least
    one ID (pmcid/pmid/doi) but are missing their title or abstract -- the same batch-lookup
    mechanism `enrich_missing_metadata` above already uses for `raw_records.csv`, applied here to
    the already-built canonical dataset instead (that file predates this dedicated step; see
    METHODS_REVIEW.md Sec 8.2 -- 428 `registry_confirmed` positives, the DOME registry API dump,
    have no abstract at all, which is both a genuine information gap and a training-data leak --
    an empty abstract is a 100%-precision positive detector on the current data). Tries pmcid
    first, then pmid, then doi for whatever's still missing after each pass. Never overwrites an
    already-populated field, only fills genuine blanks. Returns the updated DataFrame (a copy) plus
    a small stats dict: `targeted` (rows missing title/abstract), `found` (rows EPMC actually
    matched and filled), `still_missing` (targeted rows EPMC had no match for, under any of their
    IDs)."""
    df = df.copy()
    needs = (
        df["title"].apply(_blank) | df["abstract"].apply(_blank)
    )
    remaining = set(df.index[needs])
    targeted = len(remaining)
    if not remaining:
        return df, {"targeted": 0, "found": 0, "still_missing": 0}

    found_count = 0
    for id_col, id_type in (("pmcid", "pmcid"), ("pmid", "pmid"), ("doi", "doi")):
        if not remaining:
            break
        lookup: dict[str, int] = {}
        for i in remaining:
            value = df.at[i, id_col]
            if isinstance(value, str) and value.strip():
                lookup[value] = i
        if not lookup:
            continue
        if show_progress:
            print(f"  enrich-metadata: {len(lookup):,} rows to try by {id_type} "
                  f"({len(remaining):,} still missing overall)...")
        results = client.get_by_ids(list(lookup.keys()), id_type=id_type, show_progress=show_progress)
        matches = {lookup[id_value]: result for id_value, result in results.items()}
        if show_progress and matches:
            print(f"  enrich-metadata: applying {len(matches):,} matched {id_type} results...")
        _apply_epmc_results_batch(df, matches)
        remaining.difference_update(matches)
        found_count += len(matches)
        if show_progress:
            print(f"  enrich-metadata: {id_type} pass done -- {found_count:,} found so far, "
                  f"{len(remaining):,} still remaining.")

    return df, {"targeted": targeted, "found": found_count, "still_missing": len(remaining)}


def _apply_epmc_results_batch(df: pd.DataFrame, matches: dict[int, dict]) -> None:
    """Vectorized replacement for the old row-by-row `.at[]` loop -- real, measured bottleneck
    confirmed live (2026-08-27): applying ~40k matched pmcid results this way took 20+ minutes of
    silent, unlogged CPU-bound work on a memory-pressured host, dwarfing the ~7 minutes the actual
    EPMC network fetch for the same batch took. Building one small results-only DataFrame and
    writing each column via a single masked `.loc[]` assignment (one vectorized op per column, not
    one Python-level pandas cell touch per row per field) is the fix -- same "never overwrite an
    already-populated field" semantics, just done in bulk."""
    if not matches:
        return
    idx = list(matches.keys())

    def _mesh(i):
        mesh = extract_mesh_headings(matches[i])
        return json.dumps(mesh) if mesh else None

    def _pub_types(i):
        values = _non_null_strings((matches[i].get("pubTypeList") or {}).get("pubType"))
        return json.dumps(values) if values else None

    def _keywords(i):
        values = _non_null_strings((matches[i].get("keywordList") or {}).get("keyword"))
        return json.dumps(values) if values else None

    def _oa(i):
        value = matches[i].get("isOpenAccess")
        return str(value == "Y") if value in ("Y", "N") else None

    def _fulltext(i):
        result = matches[i]
        return "True" if result.get("inEPMC") == "Y" or result.get("inPMC") == "Y" else None

    new = pd.DataFrame(
        {
            "title": [matches[i].get("title") or None for i in idx],
            "abstract": [matches[i].get("abstractText") or None for i in idx],
            "journal": [
                (matches[i].get("journalInfo") or {}).get("journal", {}).get("title") for i in idx
            ],
            "authors": [matches[i].get("authorString") or None for i in idx],
            "year": [
                str(int(matches[i]["pubYear"]))
                if matches[i].get("pubYear") and str(matches[i]["pubYear"]).isdigit()
                else None
                for i in idx
            ],
            "doi": [clean_doi(matches[i].get("doi")) for i in idx],
            "pmcid": [clean_pmcid(matches[i].get("pmcid")) for i in idx],
            "pmid": [clean_pmid(matches[i].get("pmid")) for i in idx],
            "mesh_headings": [_mesh(i) for i in idx],
            "pub_types": [_pub_types(i) for i in idx],
            "keywords_author": [_keywords(i) for i in idx],
            "is_open_access": [_oa(i) for i in idx],
            "fulltext_available": [_fulltext(i) for i in idx],
        },
        index=idx,
    )
    for col in new.columns:
        blank_mask = df.loc[idx, col].apply(_blank)
        values = new.loc[idx, col]
        fill_mask = blank_mask & values.notna()
        fill_idx = fill_mask[fill_mask].index
        if len(fill_idx):
            df.loc[fill_idx, col] = values.loc[fill_idx]
    df.loc[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
