#!/usr/bin/env python3
"""SCAFFOLD -- not functional. The shape a cross-link fetcher should take; see README.md.

One subcommand per source, each: takes keys (pmid/doi/pmcid) from a corpus export or from moros
(read-only), fetches with a retrying session, streams rows to `output/<source>_links.csv` one batch
at a time (resumable), and never writes to the database. Loading is a separate, allowlisted step.

    python3 cross_links/fetch_cross_links.py epmc-datalinks --limit 100
    python3 cross_links/fetch_cross_links.py zenodo --limit 100
    python3 cross_links/fetch_cross_links.py huggingface --limit 100
    python3 cross_links/fetch_cross_links.py dome-registry --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

OUTPUT_COLUMNS = [
    "pid",            # moros _id (UUID5), the merge key
    "key_type",       # pmid | doi | pmcid -- which key produced the hit
    "key",            # the key value used
    "source",         # epmc_datalinks | zenodo | huggingface | kaggle | dome_registry | fulltext_urls
    "field",          # which identifiers.* field this would fill
    "value",          # the identifier / URL found ("" for a confirmed miss)
    "evidence",       # where it was seen (API record id, XML xpath, ...)
    "fetched_at",     # ISO-8601 UTC
]

SOURCES = ("epmc-datalinks", "zenodo", "huggingface", "kaggle", "dome-registry", "fulltext-urls")


def not_implemented(source: str) -> int:
    print(f"{source}: not implemented yet -- see cross_links/README.md for the source notes and rules.")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    for name in SOURCES:
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, default=None,
                       help="Corpus export with pid/pmid/pmcid/doi columns; default: read keys from moros (read-only).")
        p.add_argument("--output", type=Path, default=OUTPUT_DIR / f"{name.replace('-', '_')}_links.csv")
        p.add_argument("--limit", type=int, default=None, help="Seeded random sample, never a prefix.")
        p.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    return not_implemented(args.source)


if __name__ == "__main__":
    sys.exit(main())
