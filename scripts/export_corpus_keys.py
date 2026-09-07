#!/usr/bin/env python3
"""Exports every document's identifiers from moros, read-only, as the key file a citation refresh
reads: `pid, pmid, pmcid, doi`. `fetch_citations.py --input <this file> --max-age-days N` then
re-fetches only the stale keys.

Why this exists: `fetch_citations.py`'s default inputs are the original staging CSVs, which cover
only the first load. Every document added since exists solely in moros, so the key list has to
come from the live collection -- the same reason `join_citations.py` has `--corpus-from-moros`.

    python3 scripts/export_corpus_keys.py                       # -> moros_pipeline/output/corpus_keys.csv
    python3 scripts/export_corpus_keys.py --query '{"llm_classification.classification": "positive"}'
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "moros_pipeline" / "scripts"))

from moros_client import ID_FIELD, Moros  # noqa: E402

DEFAULT_OUT = REPO / "moros_pipeline" / "output" / "corpus_keys.csv"
COLUMNS = ["pid", "pmid", "pmcid", "doi"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--query", default="{}", help="Mongo filter as JSON (default: every document).")
    args = parser.parse_args()

    query = json.loads(args.query)
    projection = {ID_FIELD: 1, "identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1}
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    n = 0
    with Moros.from_env() as moros, tmp.open("w", newline="", encoding="utf-8") as f:
        print(f"export_corpus_keys: reading {moros.describe()}")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for doc in moros.iter_documents(query, projection, batch_size=5_000):
            ids = doc.get("identifiers") or {}
            writer.writerow({
                "pid": doc[ID_FIELD],
                "pmid": ids.get("pmid") or "",
                "pmcid": ids.get("pmcid") or "",
                "doi": ids.get("doi") or "",
            })
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} documents", flush=True)
    tmp.replace(args.out)
    print(f"export_corpus_keys: wrote {n:,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
