"""Bulk blunt-match candidate construction: queries ALL of Europe PMC for
`"artificial intelligence"` OR `"machine learning"` (one combined, deduplicated query -- the
search API itself returns each unique record once even if it matches both phrases -- no separate
fetch-both-then-dedupe needed for that), capturing full metadata (title/abstract/authors/journal/
year/DOI/PMID/PMCID/MeSH/pub types/open-access/author keywords) in one pass -- no separate
enrichment needed for this source.

`fetch_ai_ml_range` fetches an entire year range in one call (e.g. `dome-triage bulk-match fetch
--year-from 2000 --year-to 2026`) -- internally still one EPMC query per year (checkpointed via
per-year `.done` markers, so an interrupted multi-year run resumes at the next incomplete year,
not from scratch), with a live tqdm bar per year and a printed per-year line, so a human watching
a long multi-decade run in their own terminal sees exactly where it is. Within a single year, an
interrupted fetch is NOT resumed mid-year (only completed years are skipped on rerun) --
documented honestly rather than promising cursor-level restart this doesn't implement.

`count_ai_ml_breakdown` answers "how many mention AI, how many mention ML, how many total
(deduplicated)" via three cheap count-only queries (`EpmcClient.count`, pageSize=1,
resultType=idlist) -- no second full-metadata fetch needed just to report the breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from dome_triage.ingest.epmc_client import EpmcClient
from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid, clean_pmid
from dome_triage.ontology.mesh import extract_mesh_headings
from dome_triage.schema import RawRecord

AI_ML_QUERY = '"artificial intelligence" OR "machine learning"'


def _year_query(year: int, term_query: str = AI_ML_QUERY) -> str:
    return f"({term_query}) AND (FIRST_PDATE:[{year}-01-01 TO {year}-12-31])"


def _range_query(year_from: int, year_to: int, term_query: str = AI_ML_QUERY) -> str:
    return f"({term_query}) AND (FIRST_PDATE:[{year_from}-01-01 TO {year_to}-12-31])"


def fetch_ai_ml_candidates(client: EpmcClient, year: int, checkpoint_dir: Path) -> Path:
    """Fetches every AI/ML-matching record for `year` (resultType=core) into
    checkpoint_dir/bulk_match_<year>.jsonl. If that year was already completed (a
    `.done` marker exists), skips the fetch entirely and returns the existing path."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / f"bulk_match_{year}.jsonl"
    done_marker = checkpoint_dir / f"bulk_match_{year}.done"

    if done_marker.exists():
        return output_path

    query = _year_query(year)
    with open(output_path, "w") as f:
        for result in client.search(query, result_type="core", show_progress=True):
            f.write(json.dumps(result) + "\n")

    done_marker.write_text("complete\n")
    return output_path


def fetch_ai_ml_range(
    client: EpmcClient, year_from: int, year_to: int, checkpoint_dir: Path
) -> list[Path]:
    """Fetches every year in [year_from, year_to] (inclusive) in one call -- one EPMC query per
    year internally, reusing fetch_ai_ml_candidates' checkpoint/resume mechanism, so re-running
    after an interruption only re-fetches incomplete years. Prints a line per year so a human
    watching a long multi-year run sees live progress, not a silent block."""
    if year_to < year_from:
        raise ValueError(f"year_to ({year_to}) must be >= year_from ({year_from})")
    output_paths = []
    for year in range(year_from, year_to + 1):
        print(f"--- bulk-match fetch: {year} ---")
        output_paths.append(fetch_ai_ml_candidates(client, year, checkpoint_dir))
    return output_paths


def count_ai_ml_breakdown(client: EpmcClient, year_from: int, year_to: int) -> dict:
    """Three cheap count-only queries (no full metadata fetch) for [year_from, year_to]: how many
    records mention "artificial intelligence" alone, "machine learning" alone, and the combined
    deduplicated total that fetch_ai_ml_range will actually retrieve (EPMC's own OR-query
    semantics return each matching record exactly once even if it matches both phrases)."""
    return {
        "ai_count": client.count(_range_query(year_from, year_to, '"artificial intelligence"')),
        "ml_count": client.count(_range_query(year_from, year_to, '"machine learning"')),
        "combined_count": client.count(_range_query(year_from, year_to, AI_ML_QUERY)),
    }


def core_result_to_raw_record(
    result: dict,
    source_name: str,
    source_file: str,
    label: str = "unlabeled",
    label_confidence: str = "unscored",
) -> RawRecord:
    pub_year = result.get("pubYear")
    year = int(pub_year) if pub_year and str(pub_year).isdigit() else None
    is_open_access = result.get("isOpenAccess")

    return RawRecord(
        source_name=source_name,
        source_file=source_file,
        label=label,
        label_confidence=label_confidence,
        pmcid=clean_pmcid(result.get("pmcid")),
        pmid=clean_pmid(result.get("pmid")),
        doi=clean_doi(result.get("doi")),
        title=result.get("title") or None,
        abstract=result.get("abstractText") or None,
        journal=(result.get("journalInfo", {}) or {}).get("journal", {}).get("title"),
        authors=result.get("authorString") or None,
        year=year,
        mesh_headings=extract_mesh_headings(result),
        pub_types=_non_null_strings((result.get("pubTypeList") or {}).get("pubType")),
        is_open_access=(is_open_access == "Y") if is_open_access in ("Y", "N") else None,
        keywords_author=_non_null_strings((result.get("keywordList") or {}).get("keyword")),
        fulltext_available=bool(result.get("inEPMC") == "Y" or result.get("inPMC") == "Y"),
    )


def _non_null_strings(values: list | None) -> list[str]:
    """Real Europe PMC records occasionally have a null entry inside pubTypeList/keywordList
    (confirmed live across the 2000-2026 bulk fetch, not just a theoretical edge case) --
    RawRecord's pub_types/keywords_author are `list[str]`, so a bare null entry fails Pydantic
    validation. Filter rather than reject the whole record over one bad list element."""
    return [v for v in (values or []) if isinstance(v, str)]


def load_bulk_match_year(jsonl_path: Path, year: int) -> list[RawRecord]:
    """Loads a fetched per-year JSONL cache into RawRecords."""
    source_name = f"bulk_match_{year}"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            records.append(core_result_to_raw_record(result, source_name, str(jsonl_path)))
    return records
