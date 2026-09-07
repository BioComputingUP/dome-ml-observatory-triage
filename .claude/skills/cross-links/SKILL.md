---
name: cross-links
description: >
  Work on the reserved external-identifier fields (identifiers.dome_registry, bioai_repo,
  huggingface, kaggle, zenodo) — Europe PMC data links, software links, repository and registry
  cross-references. Trigger on "cross links", "data links", "software links", "fill the
  identifiers fields", "link papers to Zenodo / Hugging Face / the DOME Registry". This is a
  scaffold: the fetcher is a shell, no write mode exists yet, and nothing here may write to moros
  until one is added deliberately.
---

# cross-links

Read `cross_links/README.md` first: it lists what each field is meant to hold, the candidate
sources per field, and the rules.

## State

- `cross_links/fetch_cross_links.py` is an argparse shell (one subcommand per source, all
  returning "not implemented"). Its output columns (`pid, key_type, key, source, field, value,
  evidence, fetched_at`) are the contract the real fetchers keep.
- `moros_write.py::WRITE_MODES` has **no `identifiers` mode**. Until one exists, nothing from this
  folder can be written, by construction.

## When building a source

1. Implement it as one subcommand: keys from a corpus export or read-only from moros; a retrying
   `requests` session; stream rows per batch to `cross_links/output/<source>_links.csv`; resumable
   (skip keys already present); `--limit` as a seeded random sample, never a prefix.
2. Fetch by `pmid -> doi -> pmcid`, record the key and source per row, and write `""` for a
   confirmed miss (never leave `null` behind).
3. Full-text mining only where `source.access.open_access` is true, from Europe PMC's OA XML;
   never store full text.
4. Measure precision on a sample before any load, and record the numbers in the README.
5. Only then: add an `identifiers` write mode (the five leaf paths + `schema_version`) to
   `moros_write.py` with tests, a `load_fields.py`-shaped loader, and go through `moros-write`
   (dry run → `--limit` → confirm → verify).
6. If a field needs to become a list, that is a `schema.py` change: `schema-sync` first.
