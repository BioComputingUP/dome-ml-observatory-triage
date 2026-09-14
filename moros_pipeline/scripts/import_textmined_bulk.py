"""Imports Europe PMC's bulk text-mined accession dump -- zero API calls -- as a cross-check of,
or a fallback for, the annotations API.

Identifiers are written here exactly as Europe PMC returned them, punctuation and all: this is the
raw record. `build_data_links.py` (with `link_identifiers.py`) is the only place they are cleaned,
so a better normaliser never needs a re-fetch.

`https://ftp.ebi.ac.uk/pub/databases/pmc/TextMinedTerms/` holds one CSV per accession type
(`pdb.csv`, `uniprot.csv`, `doi.csv`, `gen.csv`, `nct.csv`, ...), each `accession, PMCID, EXTID,
SOURCE`, refreshed monthly. It covers the full-text-mined articles only, and on 2026-09-14 the
2026-08-31 snapshot had most per-type files at zero bytes (eight non-empty: doi 423 MB, gen 137 MB,
nct, pdb, refseq, refsnp, rrid, uniprot) -- so it is a check on the annotations API's coverage per
type, and the route to use if that API is ever throttled, not the primary source.

The join is on the record's Europe PMC identity `(SOURCE, EXTID)`, or on `PMCID`, against
`epmc_metadata.csv`. Output is the annotations JSONL shape (`provider: "ftp_bulk"`, no section,
no URI) so `build_data_links.py` reads it through the same reducer; a reference-list DOI cannot be
told apart here, so DOIs are kept only by data-repository prefix.

    python3 import_textmined_bulk.py --list                 # what the dump holds right now
    python3 import_textmined_bulk.py --types pdb,uniprot    # a subset
    python3 import_textmined_bulk.py                        # every non-empty file
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/pmc/TextMinedTerms/"
DEFAULT_DIR = FOLDER_DIR / "output" / "textmined"
DEFAULT_METADATA = FOLDER_DIR / "output" / "epmc_metadata.csv"
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "epmc_textmined_bulk.jsonl"

_HREF_RE = re.compile(r'href="([A-Za-z0-9_.-]+\.csv)"')


def list_files(session: requests.Session) -> list[tuple[str, int]]:
    """(file name, size in bytes) for every CSV in the dump, sizes from HEAD."""
    index = session.get(BASE_URL, timeout=60)
    index.raise_for_status()
    names = sorted(set(_HREF_RE.findall(index.text)))
    files = []
    for name in names:
        head = session.head(BASE_URL + name, timeout=60, allow_redirects=True)
        size = int(head.headers.get("Content-Length") or 0)
        files.append((name, size))
    return files


def download(session: requests.Session, name: str, size: int, target_dir: Path) -> Path:
    target = target_dir / name
    if target.exists() and target.stat().st_size == size:
        return target
    tmp = target.with_suffix(".tmp")
    with session.get(BASE_URL + name, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, target)
    return target


def load_identities(metadata_path: Path) -> tuple[set[tuple[str, str]], dict[str, tuple[str, str]]]:
    identities: set[tuple[str, str]] = set()
    by_pmcid: dict[str, tuple[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            source, ext_id = (row.get("epmc_source") or "").strip(), (row.get("epmc_id") or "").strip()
            if not source or not ext_id:
                continue
            identities.add((source, ext_id))
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                by_pmcid[pmcid] = (source, ext_id)
    return identities, by_pmcid


def scan(path: Path, accession_type: str, identities: set[tuple[str, str]],
         by_pmcid: dict[str, tuple[str, str]],
         found: dict[tuple[str, str], list[tuple[str, str]]]) -> int:
    n = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: <type>,PMCID,EXTID,SOURCE
        for row in reader:
            if len(row) < 4:
                continue
            accession, pmcid, ext_id, source = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            identity = (source, ext_id)
            if identity not in identities:
                identity = by_pmcid.get(pmcid)
                if identity is None:
                    continue
            found[identity].append((accession_type, accession))
            n += 1
    return n


def run(target_dir: Path, metadata_path: Path, output_path: Path, types: set[str] | None,
        list_only: bool) -> None:
    session = requests.Session()
    files = list_files(session)
    non_empty = [(n, s) for n, s in files if s > 0]
    print(f"import_textmined_bulk: {len(files)} files listed, {len(non_empty)} non-empty:")
    for name, size in non_empty:
        print(f"  {name:<22} {size / 1e6:>8.1f} MB")
    if list_only:
        return
    if types:
        non_empty = [(n, s) for n, s in non_empty if n.removesuffix(".csv") in types]
    target_dir.mkdir(parents=True, exist_ok=True)
    identities, by_pmcid = load_identities(metadata_path)
    print(f"import_textmined_bulk: {len(identities):,} corpus identities from {metadata_path.name}")

    found: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    started = time.time()
    for name, size in non_empty:
        path = download(session, name, size, target_dir)
        n = scan(path, name.removesuffix(".csv"), identities, by_pmcid, found)
        print(f"  {name}: {n:,} accessions matched {len(found):,} identities so far")

    fetched_at = datetime.now(timezone.utc).isoformat()
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for (source, ext_id), hits in found.items():
            f.write(json.dumps({
                "source": source, "id": ext_id, "fetched_at": fetched_at, "status": "ok",
                "provider": "ftp_bulk",
                "annotations": [{"exact": acc, "uri": None, "sub_type": acc_type,
                                 "provider": "ftp_bulk", "section": None, "frequency": None,
                                 "file_name": None} for acc_type, acc in hits],
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)
    print(f"import_textmined_bulk: {len(found):,} records with accessions -> {output_path} "
          f"in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--types", default=None, help="Comma-separated file stems, e.g. pdb,uniprot")
    parser.add_argument("--list", action="store_true", help="List the dump and stop.")
    args = parser.parse_args()
    run(args.dir, args.metadata, args.output,
        set(args.types.split(",")) if args.types else None, args.list)


if __name__ == "__main__":
    main()
