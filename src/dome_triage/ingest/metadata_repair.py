"""Resumable, batched, concurrent metadata repair for the bulk candidate pool via NCBI.

Design constraints this exists to satisfy (all real, all learned the hard way on 2026-08-27):

* **The API is the only thing allowed to be slow.** The previous EPMC-based backfill spent ~7
  minutes fetching and then 20+ *silent* minutes applying results one pandas cell at a time. Here
  every apply is vectorized and every network call is batched 200 ids deep and issued concurrently
  up to NCBI's published rate ceiling, so wall-clock is essentially `total_batches / rate`.
* **Nothing is lost to an interrupt.** Every batch appends to a JSONL checkpoint and flushes
  immediately, and a re-run skips whatever is already recorded -- including ids that were looked up
  and genuinely *not found*, which would otherwise be retried forever.
* **Progress is always visible.** Both phases drive a tqdm bar with a live ETA; there is no phase
  that runs without output.
* **The CSV is only ever replaced atomically**, written chunk-by-chunk to a temp file and then
  `os.replace`d, so an interrupt mid-write cannot leave a half-written 1.8GB pool behind.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from dome_triage.ingest.crossref_client import CROSSREF_BATCH_SIZE, CrossrefClient
from dome_triage.ingest.ncbi_client import BATCH_SIZE, DOI_BATCH_SIZE, NcbiClient, chunked

_READ_CHUNK_ROWS = 100_000
_MERGE_COLUMNS = ["title", "abstract", "journal", "year", "authors", "doi", "pmcid"]


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


class _Checkpoint:
    """Append-only JSONL, flushed per write so a kill -9 loses at most the in-flight batch."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = None

    def load_done(self) -> set[str]:
        if not self.path.exists():
            return set()
        done: set[str] = set()
        with self.path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line from an earlier kill -- harmless, just re-do it
                # A *transient* failure (429, timeout, connection reset) is NOT an answer. Recording
                # it as done would permanently bury ids that were merely rate-limited -- real
                # incident 2026-08-27, where Crossref 429s were being written as "not found".
                # Genuine misses (the API answered, the record isn't there) stay done forever.
                if row.get("error"):
                    continue
                if "query_id" in row:
                    done.add(row["query_id"])
        return done

    def __enter__(self):
        self._handle = self.path.open("a")
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.close()
            self._handle = None

    def write_many(self, rows: list[dict]) -> None:
        with self._lock:
            for row in rows:
                self._handle.write(json.dumps(row) + "\n")
            self._handle.flush()


def scan_missing_abstracts(csv_path: Path) -> dict:
    """Chunked scan (never loads the 1.8GB pool whole) for rows missing an abstract, bucketed by
    which identifier is available to repair them with."""
    pmids: set[str] = set()
    pmcids_needing_pmid: set[str] = set()
    dois_needing_pmid: set[str] = set()
    n_rows = 0
    n_missing = 0
    n_unrepairable = 0

    usecols = ["pmcid", "pmid", "doi", "abstract"]
    for chunk in pd.read_csv(csv_path, dtype=str, usecols=usecols, chunksize=_READ_CHUNK_ROWS):
        n_rows += len(chunk)
        missing = chunk[_blank(chunk["abstract"])]
        n_missing += len(missing)
        has_pmid = ~_blank(missing["pmid"])
        has_pmcid = ~_blank(missing["pmcid"])
        has_doi = ~_blank(missing["doi"])

        pmids.update(missing.loc[has_pmid, "pmid"].str.strip())
        pmcids_needing_pmid.update(missing.loc[~has_pmid & has_pmcid, "pmcid"].str.strip())
        dois_needing_pmid.update(missing.loc[~has_pmid & ~has_pmcid & has_doi, "doi"].str.strip())
        n_unrepairable += int((~has_pmid & ~has_pmcid & ~has_doi).sum())

    return {
        "n_rows": n_rows,
        "n_missing_abstract": n_missing,
        "pmids": sorted(pmids),
        "pmcids_needing_pmid": sorted(pmcids_needing_pmid),
        "dois_needing_pmid": sorted(dois_needing_pmid),
        "n_unrepairable": n_unrepairable,
    }


def _safe_call(call, batch: list[str], label: str):
    """One bad batch must never kill a long unattended run (real incident 2026-08-27: a single
    mixed-id-type 400 aborted the whole process, and Python's block-buffered stdout meant the
    redirected log was empty). Returns None on failure, having printed exactly what broke; the
    caller records those ids as attempted-and-not-found so a resume moves past them."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: never abort the run
        print(f"  metadata-repair: {label} batch of {len(batch)} failed ({exc}) -- continuing.")
        return None


def _run_batches(client: NcbiClient, batches: list[list[str]], worker, desc: str, checkpoint):
    """Shared concurrent driver. Worker count is deliberately a small multiple of the rate ceiling:
    the shared token bucket inside NcbiClient sets the real pace, extra threads only exist to keep
    the pipe full while a slow response is in flight."""
    if not batches:
        return 0
    n_workers = max(2, min(16, int(client.rate_per_sec * 2)))
    found = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker, batch): batch for batch in batches}
        with tqdm(total=sum(len(b) for b in batches), desc=desc, unit="id") as bar:
            for future in as_completed(futures):
                batch = futures[future]
                rows = future.result()
                checkpoint.write_many(rows)
                found += sum(1 for row in rows if row.get("found"))
                bar.update(len(batch))
                bar.set_postfix(found=found)
    return found


def fetch_repairs(
    csv_path: Path,
    records_checkpoint: Path,
    idmap_checkpoint: Path,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """Phase 1 (id conversion, only for rows with no pmid) + phase 2 (PubMed efetch). Both fully
    resumable; returns real counts, no estimates."""
    scan = scan_missing_abstracts(csv_path)
    print(
        f"metadata-repair: {scan['n_rows']:,} rows -- {scan['n_missing_abstract']:,} missing abstract.\n"
        f"  have pmid directly:            {len(scan['pmids']):,}\n"
        f"  need pmcid -> pmid conversion: {len(scan['pmcids_needing_pmid']):,}\n"
        f"  need doi   -> pmid conversion: {len(scan['dois_needing_pmid']):,}\n"
        f"  no identifier at all (skipped): {scan['n_unrepairable']:,}"
    )

    client = NcbiClient(api_key=api_key)
    print(
        f"metadata-repair: NCBI rate ceiling {client.rate_per_sec:.0f} req/s "
        f"({'API key found' if client.api_key else 'no API key -- set NCBI_API_KEY for 10/s'}), "
        f"{BATCH_SIZE} ids per request."
    )

    try:
        # ---- phase 1: convert whatever has no pmid -------------------------------------------
        idmap_cp = _Checkpoint(idmap_checkpoint)
        already_converted = idmap_cp.load_done()

        def convert_worker(batch: list[str]) -> list[dict]:
            mapped = _safe_call(lambda: client.convert_ids(batch), batch, "convert")
            if mapped is None:
                return [{"query_id": value, "found": False, "error": True} for value in batch]
            return [
                {"query_id": value, "found": value in mapped, **(mapped.get(value) or {})}
                for value in batch
            ]

        # PMCIDs and DOIs must NOT share a batch -- the converter 400s on a mixed batch (real,
        # confirmed 2026-08-27), so each type gets its own pass and its own batch size.
        convert_passes = [
            ("pmcid", scan["pmcids_needing_pmid"], BATCH_SIZE),
            ("doi", scan["dois_needing_pmid"], DOI_BATCH_SIZE),
        ]
        any_converted = False
        with idmap_cp as cp:
            for id_type, values, batch_size in convert_passes:
                pending = [value for value in values if value not in already_converted]
                if not pending:
                    continue
                any_converted = True
                _run_batches(
                    client,
                    list(chunked(pending, size=batch_size)),
                    convert_worker,
                    f"convert {id_type} -> pmid",
                    cp,
                )
        if not any_converted:
            print(f"metadata-repair: id conversion already complete ({len(already_converted):,} done).")

        # Everything converted so far (this run + any earlier run) feeds phase 2.
        converted_pmids: set[str] = set()
        if idmap_checkpoint.exists():
            with idmap_checkpoint.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("found") and row.get("pmid"):
                        converted_pmids.add(str(row["pmid"]))

        # ---- phase 2: efetch abstracts -------------------------------------------------------
        records_cp = _Checkpoint(records_checkpoint)
        already_fetched = records_cp.load_done()
        all_pmids = sorted(set(scan["pmids"]) | converted_pmids)
        to_fetch = [pmid for pmid in all_pmids if pmid not in already_fetched]
        if limit is not None:
            to_fetch = to_fetch[:limit]

        print(
            f"metadata-repair: {len(all_pmids):,} pmids in scope, "
            f"{len(already_fetched):,} already done, {len(to_fetch):,} to fetch."
        )

        def fetch_worker(batch: list[str]) -> list[dict]:
            fetched = _safe_call(lambda: client.efetch_pubmed(batch), batch, "efetch")
            if fetched is None:
                return [{"query_id": pmid, "found": False, "error": True} for pmid in batch]
            return [
                {"query_id": pmid, "found": pmid in fetched, **(fetched.get(pmid) or {})}
                for pmid in batch
            ]

        found = 0
        if to_fetch:
            with records_cp as cp:
                found = _run_batches(
                    client, list(chunked(to_fetch)), fetch_worker, "efetch pubmed", cp
                )
        else:
            print("metadata-repair: nothing left to fetch.")
    finally:
        client.close()

    return {
        "n_missing_abstract": scan["n_missing_abstract"],
        "n_pmids_in_scope": len(all_pmids),
        "n_fetched_this_run": len(to_fetch),
        "n_found_this_run": found,
        "n_unrepairable": scan["n_unrepairable"],
    }


def scan_incomplete_ids(csv_path: Path) -> dict:
    """Rows holding at least one identifier but not all three, bucketed by the best identifier to
    ask the converter with (pmid is the most reliable key, then pmcid, then doi)."""
    by_pmid: set[str] = set()
    by_pmcid: set[str] = set()
    by_doi: set[str] = set()
    n_rows = 0
    n_complete = 0
    n_no_id = 0

    usecols = ["pmcid", "pmid", "doi"]
    for chunk in pd.read_csv(csv_path, dtype=str, usecols=usecols, chunksize=_READ_CHUNK_ROWS):
        n_rows += len(chunk)
        has_pmid = ~_blank(chunk["pmid"])
        has_pmcid = ~_blank(chunk["pmcid"])
        has_doi = ~_blank(chunk["doi"])
        complete = has_pmid & has_pmcid & has_doi
        none_at_all = ~has_pmid & ~has_pmcid & ~has_doi
        n_complete += int(complete.sum())
        n_no_id += int(none_at_all.sum())

        need = ~complete & ~none_at_all
        by_pmid.update(chunk.loc[need & has_pmid, "pmid"].str.strip())
        by_pmcid.update(chunk.loc[need & ~has_pmid & has_pmcid, "pmcid"].str.strip())
        by_doi.update(chunk.loc[need & ~has_pmid & ~has_pmcid & has_doi, "doi"].str.strip())

    return {
        "n_rows": n_rows,
        "n_complete": n_complete,
        "n_no_id": n_no_id,
        "by_pmid": sorted(by_pmid),
        "by_pmcid": sorted(by_pmcid),
        "by_doi": sorted(by_doi),
    }


def complete_ids(csv_path: Path, checkpoint: Path, limit: Optional[int] = None) -> dict:
    """Fills in the missing members of the pmid/pmcid/doi triple for every row that has at least
    one of them, via NCBI's ID converter. Resumable and batched exactly like the abstract repair;
    each identifier type gets its own pass because the converter rejects a mixed batch."""
    scan = scan_incomplete_ids(csv_path)
    to_do = len(scan["by_pmid"]) + len(scan["by_pmcid"]) + len(scan["by_doi"])
    print(
        f"complete-ids: {scan['n_rows']:,} rows -- {scan['n_complete']:,} already have all three, "
        f"{scan['n_no_id']:,} have none (unfixable), {to_do:,} to look up.\n"
        f"  query by pmid:  {len(scan['by_pmid']):,}\n"
        f"  query by pmcid: {len(scan['by_pmcid']):,}\n"
        f"  query by doi:   {len(scan['by_doi']):,}"
    )

    client = NcbiClient()
    print(
        f"complete-ids: NCBI rate ceiling {client.rate_per_sec:.0f} req/s "
        f"({'API key found' if client.api_key else 'no API key -- set NCBI_API_KEY for 10/s'})."
    )
    cp = _Checkpoint(checkpoint)
    already = cp.load_done()

    def worker(batch: list[str]) -> list[dict]:
        mapped = _safe_call(lambda: client.convert_ids(batch), batch, "convert")
        if mapped is None:
            return [{"query_id": value, "found": False, "error": True} for value in batch]
        return [
            {"query_id": value, "found": value in mapped, **(mapped.get(value) or {})}
            for value in batch
        ]

    found = 0
    try:
        with cp as handle:
            for id_type, values, size in (
                ("pmid", scan["by_pmid"], BATCH_SIZE),
                ("pmcid", scan["by_pmcid"], BATCH_SIZE),
                ("doi", scan["by_doi"], DOI_BATCH_SIZE),
            ):
                pending = [value for value in values if value not in already]
                if limit is not None:
                    pending = pending[:limit]
                if not pending:
                    continue
                found += _run_batches(
                    client, list(chunked(pending, size=size)), worker, f"complete by {id_type}", handle
                )
    finally:
        client.close()
    return {"looked_up": to_do, "resolved_this_run": found}


def merge_ids(csv_path: Path, checkpoint: Path) -> dict:
    """Applies completed identifiers into the pool. Fills blanks only -- an identifier already in
    the pool is never replaced by the converter's version."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"No id checkpoint at {checkpoint} -- run the lookup first.")

    lookup: dict[str, dict] = {}
    with checkpoint.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("found"):
                continue
            payload = {key: row.get(key) for key in ("pmid", "pmcid", "doi")}
            if row.get("query_id"):
                lookup[str(row["query_id"])] = payload

    print(f"complete-ids: {len(lookup):,} resolved identifier sets to merge.")

    tmp_path = csv_path.with_suffix(".ids.tmp.csv")
    filled = Counter()
    header = True
    with tqdm(desc="merging ids", unit="row") as bar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=_READ_CHUNK_ROWS):
            # Match on whichever identifier the row already has -- that is what it was queried by.
            resolved = None
            for key in ("pmid", "pmcid", "doi"):
                candidate = chunk[key].map(
                    lambda v: lookup.get(str(v).strip()) if pd.notna(v) else None
                )
                resolved = candidate if resolved is None else resolved.where(resolved.notna(), candidate)
            hits = resolved[resolved.notna()]
            if len(hits):
                for col in ("pmid", "pmcid", "doi"):
                    values = hits.map(lambda payload: payload.get(col))
                    fill_mask = _blank(chunk.loc[hits.index, col]) & values.notna()
                    if fill_mask.any():
                        fill_idx = fill_mask[fill_mask].index
                        chunk.loc[fill_idx, col] = values.loc[fill_idx]
                        filled[col] += len(fill_idx)
            chunk.to_csv(tmp_path, mode="w" if header else "a", header=header, index=False)
            header = False
            bar.update(len(chunk))

    os.replace(tmp_path, csv_path)
    print(
        f"complete-ids: merged -- filled {filled['pmid']:,} pmid, {filled['pmcid']:,} pmcid, "
        f"{filled['doi']:,} doi."
    )
    return {"filled_pmid": filled["pmid"], "filled_pmcid": filled["pmcid"], "filled_doi": filled["doi"]}


def scan_missing_abstract_dois(csv_path: Path) -> list[str]:
    """DOIs of rows that still have no abstract -- the Crossref recovery pass's input."""
    dois: set[str] = set()
    for chunk in pd.read_csv(
        csv_path, dtype=str, usecols=["doi", "abstract"], chunksize=_READ_CHUNK_ROWS
    ):
        missing = chunk[_blank(chunk["abstract"]) & ~_blank(chunk["doi"])]
        dois.update(missing["doi"].str.strip())
    return sorted(dois)


def fetch_crossref_abstracts(
    csv_path: Path, checkpoint: Path, limit: Optional[int] = None, concurrency: int = 40
) -> dict:
    """Last-resort abstract recovery from Crossref for DOIs PubMed had no abstract for. No batch
    endpoint exists, so this is concurrent single-DOI GETs paced by Crossref's own advertised rate
    headers; resumable and checkpointed exactly like the other passes."""
    dois = scan_missing_abstract_dois(csv_path)
    cp = _Checkpoint(checkpoint)
    already = cp.load_done()
    pending = [doi for doi in dois if doi not in already]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"crossref: {len(dois):,} abstract-less rows have a doi, {len(already):,} already tried, "
        f"{len(pending):,} to try now."
    )
    if not pending:
        return {"tried": 0, "found": 0}

    client = CrossrefClient()
    print(
        f"crossref: starting at {client.limiter.rate_per_sec:.0f} req/s with {concurrency} workers"
        f"{' (polite pool)' if client.mailto else ' -- set CROSSREF_MAILTO for the faster polite pool'}"
        ", adapting to Crossref's own rate headers."
    )

    found = 0
    try:
        with cp as handle:
            batches = list(chunked(pending, size=CROSSREF_BATCH_SIZE))
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_crossref_worker, client, batch): batch for batch in batches}
                with tqdm(total=len(pending), desc="crossref", unit="doi") as bar:
                    for future in as_completed(futures):
                        rows = future.result()
                        handle.write_many(rows)
                        found += sum(1 for row in rows if row.get("found"))
                        bar.update(len(futures[future]))
                        bar.set_postfix(found=found, rate=f"{client.limiter.rate_per_sec:.1f}req/s")
    finally:
        client.close()
    return {"tried": len(pending), "found": found}


def _crossref_worker(client: CrossrefClient, batch: list[str]) -> list[dict]:
    try:
        found = client.get_abstracts_batch(batch)
    except Exception:  # noqa: BLE001 -- transient: flagged so a resume retries the whole batch
        return [{"query_id": doi, "found": False, "error": True} for doi in batch]
    # Anything absent from a successful response is a real answer -- Crossref either does not know
    # the DOI or has no abstract for it. Recorded as done so a resume never asks again.
    return [
        {"query_id": doi, "found": doi in found, **(found.get(doi) or {})} for doi in batch
    ]


def merge_crossref(csv_path: Path, checkpoint: Path) -> dict:
    """Applies Crossref-recovered abstracts into the pool, matching on DOI. Blanks only."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"No crossref checkpoint at {checkpoint} -- run the fetch first.")

    by_doi: dict[str, dict] = {}
    with checkpoint.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("found") and row.get("abstract"):
                by_doi[str(row["query_id"])] = {
                    col: row.get(col) for col in ("title", "abstract", "journal", "year")
                }

    print(f"crossref: {len(by_doi):,} recovered abstracts to merge.")
    tmp_path = csv_path.with_suffix(".crossref.tmp.csv")
    filled = 0
    header = True
    with tqdm(desc="merging crossref", unit="row") as bar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=_READ_CHUNK_ROWS):
            need = _blank(chunk["abstract"])
            if need.any():
                idx = chunk.index[need]
                resolved = chunk.loc[idx, "doi"].map(
                    lambda v: by_doi.get(str(v).strip()) if pd.notna(v) else None
                )
                hits = resolved[resolved.notna()]
                if len(hits):
                    for col in ("title", "abstract", "journal", "year"):
                        values = hits.map(lambda payload: payload.get(col))
                        fill_mask = _blank(chunk.loc[hits.index, col]) & values.notna()
                        if fill_mask.any():
                            fill_idx = fill_mask[fill_mask].index
                            chunk.loc[fill_idx, col] = values.loc[fill_idx]
                    filled += len(hits)
            chunk.to_csv(tmp_path, mode="w" if header else "a", header=header, index=False)
            header = False
            bar.update(len(chunk))

    os.replace(tmp_path, csv_path)
    print(f"crossref: merged -- {filled:,} rows repaired.")
    return {"rows_repaired": filled}


def annotate_provenance(
    csv_path: Path, records_cp: Path, crossref_cp: Path, idmap_cp: Path, idcomplete_cp: Path
) -> dict:
    """Stamps per-record provenance for the repair passes into the pool itself.

    Why this is sound rather than guesswork: every repair checkpoint only ever contains ids drawn
    from rows that were **blank when that pass scanned**, and the passes ran in a fixed order
    (Europe PMC bulk fetch -> NCBI PubMed -> Crossref). So a record appearing in the PubMed
    checkpoint *with* an abstract must have had that abstract filled by PubMed; if PubMed had left
    it blank it would then appear in the Crossref checkpoint instead. Anything holding an abstract
    that appears in neither came from the original Europe PMC fetch.

    Writes two columns:
      * `abstract_source` -- europepmc | pubmed | crossref | (blank if still no abstract)
      * `metadata_repair_sources` -- ';'-joined list of external sources that filled ANY field on
        this row (ncbi_idconv, pubmed, crossref); blank means untouched by repair."""
    def _load(path: Path, require_abstract: bool) -> set[str]:
        if not path.exists():
            return set()
        out: set[str] = set()
        with path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("found"):
                    continue
                if require_abstract and not row.get("abstract"):
                    continue
                if row.get("query_id"):
                    out.add(str(row["query_id"]))
        return out

    pubmed_abs = _load(records_cp, True)
    crossref_abs = _load(crossref_cp, True)
    pubmed_any = _load(records_cp, False)
    idconv_any = _load(idmap_cp, False) | _load(idcomplete_cp, False)
    print(
        f"annotate-provenance: pubmed abstracts={len(pubmed_abs):,}, "
        f"crossref abstracts={len(crossref_abs):,}, pubmed-touched={len(pubmed_any):,}, "
        f"idconv-touched={len(idconv_any):,}"
    )

    tmp_path = csv_path.with_suffix(".prov.tmp.csv")
    counts = Counter()
    header = True
    with tqdm(desc="annotating", unit="row") as bar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=_READ_CHUNK_ROWS):
            pmid = chunk["pmid"].fillna("").str.strip()
            pmcid = chunk["pmcid"].fillna("").str.strip()
            doi = chunk["doi"].fillna("").str.strip()
            has_abs = ~_blank(chunk["abstract"])

            from_crossref = doi.isin(crossref_abs) & has_abs
            from_pubmed = pmid.isin(pubmed_abs) & has_abs & ~from_crossref
            source = pd.Series("", index=chunk.index, dtype=object)
            source[has_abs] = "europepmc"
            source[from_pubmed] = "pubmed"
            source[from_crossref] = "crossref"
            chunk["abstract_source"] = source
            counts.update(source.value_counts().to_dict())

            touched = pd.Series("", index=chunk.index, dtype=object)
            any_id = pmid.isin(idconv_any) | pmcid.isin(idconv_any) | doi.isin(idconv_any)
            parts = [
                ("ncbi_idconv", any_id),
                ("pubmed", pmid.isin(pubmed_any)),
                ("crossref", doi.isin(crossref_abs)),
            ]
            for name, mask in parts:
                touched = touched.where(~mask, touched.str.cat([name] * len(touched), sep=";"))
            chunk["metadata_repair_sources"] = touched.str.strip(";")

            chunk.to_csv(tmp_path, mode="w" if header else "a", header=header, index=False)
            header = False
            bar.update(len(chunk))

    os.replace(tmp_path, csv_path)
    print("annotate-provenance: abstract_source breakdown --")
    for key in ("europepmc", "pubmed", "crossref", ""):
        label = key or "(no abstract)"
        print(f"  {label:16s} {counts.get(key, 0):>9,}")
    return dict(counts)


def report_completeness(csv_path: Path) -> dict:
    """Final real state of the pool: identifier coverage plus title/abstract gaps."""
    c = Counter()
    n_rows = 0
    for chunk in pd.read_csv(
        csv_path, dtype=str, usecols=["pmcid", "pmid", "doi", "title", "abstract"],
        chunksize=_READ_CHUNK_ROWS,
    ):
        n_rows += len(chunk)
        has_pmid = ~_blank(chunk["pmid"])
        has_pmcid = ~_blank(chunk["pmcid"])
        has_doi = ~_blank(chunk["doi"])
        c["has_pmid"] += int(has_pmid.sum())
        c["has_pmcid"] += int(has_pmcid.sum())
        c["has_doi"] += int(has_doi.sum())
        c["all_three"] += int((has_pmid & has_pmcid & has_doi).sum())
        c["no_id"] += int((~has_pmid & ~has_pmcid & ~has_doi).sum())
        c["missing_title"] += int(_blank(chunk["title"]).sum())
        c["missing_abstract"] += int(_blank(chunk["abstract"]).sum())
        c["missing_both"] += int((_blank(chunk["title"]) & _blank(chunk["abstract"])).sum())

    print(f"\n=== bulk pool completeness: {n_rows:,} rows ===")
    for key in ("has_pmid", "has_pmcid", "has_doi", "all_three", "no_id"):
        print(f"  {key:16s} {c[key]:>9,}  ({c[key] / n_rows:6.1%})")
    print(f"  {'missing_title':16s} {c['missing_title']:>9,}  ({c['missing_title'] / n_rows:6.1%})")
    print(f"  {'missing_abstract':16s} {c['missing_abstract']:>9,}  ({c['missing_abstract'] / n_rows:6.1%})")
    print(f"  {'missing_both':16s} {c['missing_both']:>9,}  ({c['missing_both'] / n_rows:6.1%})")
    return {"n_rows": n_rows, **dict(c)}


def merge_repairs(csv_path: Path, records_checkpoint: Path) -> dict:
    """Applies the checkpointed records into the pool. Chunked read + chunked write to a temp file
    then a single atomic `os.replace` -- an interrupt can never leave a partial pool behind. Only
    genuinely blank fields are filled; nothing already populated is overwritten."""
    if not records_checkpoint.exists():
        raise FileNotFoundError(f"No repair checkpoint at {records_checkpoint} -- run the fetch first.")

    by_pmid: dict[str, dict] = {}
    by_pmcid: dict[str, dict] = {}
    with records_checkpoint.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Deliberately NOT gated on the abstract: PubMed also hands back a pmcid/doi/journal
            # for records that simply have no abstract, and an earlier version threw all of that
            # away (real gap, 2026-08-27 -- 40,052 records' identifiers discarded for free).
            if not row.get("found"):
                continue
            payload = {col: row.get(col) for col in _MERGE_COLUMNS}
            if row.get("query_id"):
                by_pmid[str(row["query_id"])] = payload
            if row.get("pmcid"):
                by_pmcid[str(row["pmcid"])] = payload

    print(f"metadata-repair: {len(by_pmid):,} repaired records available to merge.")

    tmp_path = csv_path.with_suffix(".repair.tmp.csv")
    filled = 0
    other_fields = 0
    rows_out = 0
    header = True

    with tqdm(desc="merging", unit="row") as bar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=_READ_CHUNK_ROWS):
            need = _blank(chunk["abstract"])
            if need.any():
                idx = chunk.index[need]
                keys = chunk.loc[idx, "pmid"].map(
                    lambda v: by_pmid.get(str(v).strip()) if pd.notna(v) else None
                )
                fallback = chunk.loc[idx, "pmcid"].map(
                    lambda v: by_pmcid.get(str(v).strip()) if pd.notna(v) else None
                )
                resolved = keys.where(keys.notna(), fallback)
                hits = resolved[resolved.notna()]
                if len(hits):
                    for col in _MERGE_COLUMNS:
                        values = hits.map(lambda payload: payload.get(col))
                        target = chunk.loc[hits.index, col]
                        fill_mask = _blank(target) & values.notna()
                        if fill_mask.any():
                            fill_idx = fill_mask[fill_mask].index
                            chunk.loc[fill_idx, col] = values.loc[fill_idx]
                            # "Repaired" must mean an abstract actually landed -- matching a record
                            # whose PubMed entry has no abstract fills its ids and nothing else,
                            # and counting that as a repair would overstate the result.
                            if col == "abstract":
                                filled += len(fill_idx)
                            else:
                                other_fields += len(fill_idx)
            chunk.to_csv(tmp_path, mode="w" if header else "a", header=header, index=False)
            header = False
            rows_out += len(chunk)
            bar.update(len(chunk))

    os.replace(tmp_path, csv_path)

    still_missing = 0
    for chunk in pd.read_csv(csv_path, dtype=str, usecols=["abstract"], chunksize=_READ_CHUNK_ROWS):
        still_missing += int(_blank(chunk["abstract"]).sum())

    print(
        f"metadata-repair: merged -- {filled:,} abstracts recovered, {other_fields:,} other blank "
        f"fields (ids/journal/year) filled, {rows_out:,} rows written, "
        f"{still_missing:,} still missing an abstract."
    )
    return {
        "rows_repaired": filled,
        "other_fields_filled": other_fields,
        "rows_written": rows_out,
        "still_missing": still_missing,
    }
