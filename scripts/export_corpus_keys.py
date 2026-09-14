#!/usr/bin/env python3
"""Exports every document's identifiers from moros, read-only, as the key file a citation refresh
reads: `pid, pmid, pmcid, doi`. `fetch_citations.py --input <this file> --max-age-days N` then
re-fetches only the stale keys.

Four more columns ride along for the Europe PMC metadata / data-links passes, appended so a reader
of the first four is unaffected: `is_preprint` (True/False, from `content_filters.pub_types` --
`fetch_epmc_metadata.py` keys a preprint by DOI under `SRC:PPR`, never by pmid), `classification`,
and the Europe PMC identity `epmc_source` / `epmc_id` once the preprints backfill has populated it
(blank until then), so a later data-links run can address `/{source}/{id}` without a metadata pass.

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
COLUMNS = ["pid", "pmid", "pmcid", "doi", "is_preprint", "classification", "epmc_source", "epmc_id"]

# Two records carry the lowercase spelling (docs/preprint.md, "Selection").
PREPRINT_PUB_TYPES = {"Preprint", "preprint"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--query", default="{}", help="Mongo filter as JSON (default: every document).")
    args = parser.parse_args()

    query = json.loads(args.query)
    projection = {ID_FIELD: 1, "identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1,
                  "identifiers.epmc_id": 1, "source.epmc_source": 1,
                  "content_filters.pub_types": 1, "llm_classification.classification": 1}
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    n = 0
    with Moros.from_env() as moros, tmp.open("w", newline="", encoding="utf-8") as f:
        print(f"export_corpus_keys: reading {moros.describe()}")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for doc in moros.iter_documents(query, projection, batch_size=5_000):
            ids = doc.get("identifiers") or {}
            pub_types = (doc.get("content_filters") or {}).get("pub_types") or []
            writer.writerow({
                "pid": doc[ID_FIELD],
                "pmid": ids.get("pmid") or "",
                "pmcid": ids.get("pmcid") or "",
                "doi": ids.get("doi") or "",
                "is_preprint": "True" if PREPRINT_PUB_TYPES & set(pub_types) else "False",
                "classification": (doc.get("llm_classification") or {}).get("classification") or "",
                "epmc_source": (doc.get("source") or {}).get("epmc_source") or "",
                "epmc_id": ids.get("epmc_id") or "",
            })
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} documents", flush=True)
    tmp.replace(args.out)
    print(f"export_corpus_keys: wrote {n:,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
