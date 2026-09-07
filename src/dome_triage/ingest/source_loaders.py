"""The 4 loader adapters that the 7 configured label_sources collapse into (see
configs/sources.yaml). Every loader returns (records, unresolved) -- `unresolved` holds raw
entries that couldn't be given a usable id and must route to unresolved_needs_id_lookup.csv
rather than being silently dropped (only Adapter D produces these in practice, confirmed: 3
entries in dome_api_batches.json lack both a usable pmid and doi).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid, clean_pmid
from dome_triage.schema import RawRecord

LoaderResult = tuple[list[RawRecord], list[dict]]


def _parse_numeric_id_string(value: object) -> int | None:
    """Parse Year/Citation_Count columns stored as pandas float-cast strings (e.g. "2022.0")."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", ""):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _clean_str(value: object) -> str | None:
    """pandas turns genuinely-empty cells into NaN (a float) even under dtype=str -- and NaN is
    truthy in Python, so a plain `value or None` doesn't catch it. Every optional string field
    read from a CSV must go through this."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_match_metadata(row: pd.Series) -> dict | None:
    """The `matches` column is a Python-literal string (single-quoted dicts), not JSON."""
    raw_matches = row.get("matches")
    matches = None
    if isinstance(raw_matches, str) and raw_matches.strip():
        try:
            matches = ast.literal_eval(raw_matches)
        except (ValueError, SyntaxError):
            matches = None
    metadata = {
        "matches": matches,
        "match_count": _parse_numeric_id_string(row.get("match_count")),
        "total_weight": row.get("total_weight"),
        "match_percentage": row.get("match_percentage"),
        "top20_in_journal": row.get("Top20_In_Journal"),
    }
    return {k: v for k, v in metadata.items() if v is not None}


# --- Adapter A: curated_csv --------------------------------------------------------------------


def load_curated_csv(source_cfg: dict) -> LoaderResult:
    path = Path(source_cfg["path"])
    df = pd.read_csv(path, dtype=str, keep_default_na=True)

    records = [
        RawRecord(
            source_name=source_cfg["name"],
            source_file=str(path),
            label=source_cfg["label"],
            label_confidence=source_cfg["label_confidence"],
            pmcid=clean_pmcid(row.get("PMCID")),
            pmid=clean_pmid(row.get("PMID")),
            doi=clean_doi(row.get("DOI")),
            title=_clean_str(row.get("Title")),
            abstract=_clean_str(row.get("Abstract")),
            journal=_clean_str(row.get("Journal")),
            authors=_clean_str(row.get("Authors")),
            year=_parse_numeric_id_string(row.get("Year")),
            citation_count=_parse_numeric_id_string(row.get("Citation_Count")),
            match_metadata=_parse_match_metadata(row) or None,
        )
        for _, row in df.iterrows()
    ]
    return records, []


# --- Adapter B: id_pair_only --------------------------------------------------------------------


def load_id_pair_only(source_cfg: dict) -> LoaderResult:
    path = Path(source_cfg["path"])
    delimiter = source_cfg.get("delimiter", ",")
    df = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=True)

    records = [
        RawRecord(
            source_name=source_cfg["name"],
            source_file=str(path),
            label=source_cfg["label"],
            label_confidence=source_cfg["label_confidence"],
            pmcid=clean_pmcid(row.get("PMCID")),
            pmid=clean_pmid(row.get("PMID")),
        )
        for _, row in df.iterrows()
    ]
    return records, []


# --- Adapter C: pdf_directory_gold -----------------------------------------------------------


def _pmcids_from_flat_main_pdf(root: Path) -> list[str]:
    return sorted({p.name.split("_main.pdf")[0] for p in root.glob("PMC*_main.pdf")})


def _pmcids_from_pmcid_subdir(root: Path) -> list[str]:
    return sorted({p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("PMC")})


_PDF_LAYOUTS: dict[str, Callable[[Path], list[str]]] = {
    "flat_main_pdf": _pmcids_from_flat_main_pdf,
    "pmcid_subdir": _pmcids_from_pmcid_subdir,
}


def load_pdf_directory_gold(source_cfg: dict) -> LoaderResult:
    root = Path(source_cfg["path"])
    layout = source_cfg["pdf_layout"]
    pmcids = _PDF_LAYOUTS[layout](root)

    records = [
        RawRecord(
            source_name=source_cfg["name"],
            source_file=str(root),
            label=source_cfg["label"],
            label_confidence=source_cfg["label_confidence"],
            pmcid=clean_pmcid(pmcid),
            fulltext_available=True,
            fulltext_source_root=source_cfg["name"],
        )
        for pmcid in pmcids
    ]
    return records, []


# --- Adapter D: dome_registry_api_json ----------------------------------------------------------


def load_dome_registry_api_json(source_cfg: dict) -> LoaderResult:
    path = Path(source_cfg["path"])
    with open(path) as f:
        batches: dict[str, list[dict]] = json.load(f)

    seen_ids: set[str] = set()
    records: list[RawRecord] = []
    unresolved: list[dict] = []

    for batch_name, entries in batches.items():
        for entry in entries:
            entry_id = entry.get("_id")
            if entry_id is not None and entry_id in seen_ids:
                continue  # batches are cumulative/overlapping -- confirmed, dedupe by _id
            if entry_id is not None:
                seen_ids.add(entry_id)

            pub = entry.get("publication", {})
            pmid = clean_pmid(pub.get("pmid"))
            doi = clean_doi(pub.get("doi"))

            if not pmid and not doi:
                unresolved.append(
                    {"source_name": source_cfg["name"], "source_batch": batch_name, **entry}
                )
                continue

            records.append(
                RawRecord(
                    source_name=source_cfg["name"],
                    source_file=str(path),
                    label=source_cfg["label"],
                    label_confidence=source_cfg["label_confidence"],
                    pmid=pmid,
                    doi=doi,
                    title=pub.get("title") or None,
                    journal=pub.get("journal") or None,
                    authors=pub.get("authors") or None,
                    year=_parse_numeric_id_string(pub.get("year")),
                )
            )
    return records, unresolved


# --- Dispatch ------------------------------------------------------------------------------------

_ADAPTERS: dict[str, Callable[[dict], LoaderResult]] = {
    "curated_csv": load_curated_csv,
    "id_pair_only": load_id_pair_only,
    "pdf_directory_gold": load_pdf_directory_gold,
    "dome_registry_api_json": load_dome_registry_api_json,
}


def load_all_sources(sources_cfg: dict) -> LoaderResult:
    all_records: list[RawRecord] = []
    all_unresolved: list[dict] = []
    for source_cfg in sources_cfg["label_sources"]:
        loader = _ADAPTERS[source_cfg["adapter"]]
        records, unresolved = loader(source_cfg)
        all_records.extend(records)
        all_unresolved.extend(unresolved)
    return all_records, all_unresolved


# --- RawRecord <-> DataFrame round-tripping (data/interim/raw_records*.csv) --------------------


_RAW_RECORD_JSON_FIELDS = ("match_metadata", "mesh_headings", "pub_types", "keywords_author")


def raw_records_to_dataframe(records: list[RawRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = record.model_dump()
        for field in _RAW_RECORD_JSON_FIELDS:
            if row.get(field):
                row[field] = json.dumps(row[field])
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_raw_records(df: pd.DataFrame) -> list[RawRecord]:
    records = []
    for _, row in df.iterrows():
        data = row.to_dict()
        for field in _RAW_RECORD_JSON_FIELDS:
            if isinstance(data.get(field), str) and data[field]:
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    data[field] = None
        for key, value in list(data.items()):
            if isinstance(value, float) and pd.isna(value):
                data[key] = None
            elif isinstance(value, str) and value == "":
                data[key] = None
        if isinstance(data.get("fulltext_available"), str):
            data["fulltext_available"] = data["fulltext_available"].lower() == "true"
        if isinstance(data.get("is_open_access"), str):
            data["is_open_access"] = data["is_open_access"].lower() == "true"
        records.append(RawRecord(**data))
    return records
