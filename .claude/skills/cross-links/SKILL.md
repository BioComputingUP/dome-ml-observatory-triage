---
name: cross-links
description: >
  Work on the reserved external-identifier fields (identifiers.dome_registry, bioai_repo,
  huggingface, kaggle, zenodo): deriving them from the data_links group, and the repository and
  registry APIs Europe PMC does not cover (DOME Registry, Hugging Face, Kaggle, GitHub). Trigger
  on "cross links", "fill the identifiers fields", "link papers to Zenodo / Hugging Face / the
  DOME Registry", "software links". For Europe PMC data links themselves use the `data-links`
  skill. This is a scaffold: no identifiers write mode exists yet.
---

# cross-links

Read `cross_links/README.md` first: it lists what each field is meant to hold, the candidate
sources per field, and the rules.

## State

- Europe PMC data links are fetched and stored per document in `data_links` (schema v1.4.0) by
  the `data-links` skill. `identifiers.zenodo` and `identifiers.bioai_repo` should be *derived*
  from those links (`pid_data_links.csv`, resources `zenodo`, `github`, `software_heritage`),
  not fetched again.
- `cross_links/fetch_cross_links.py` is an argparse shell (one subcommand per source, all
  returning "not implemented"). Its output columns (`pid, key_type, key, source, field, value,
  evidence, fetched_at`) are the contract the real fetchers keep; `epmc-datalinks` becomes the
  derivation from `pid_data_links.csv`.
- `moros_write.py::WRITE_MODES` has **no `identifiers` mode**, and the `data_links` mode cannot
  reach `identifiers.*`. Until an identifiers mode exists, nothing from this folder can be written,
  by construction.

## When building a source

1. Implement it as one subcommand: keys from a corpus export or read-only from moros; a retrying
   `requests` session at high concurrency; stream rows per batch to
   `cross_links/output/<source>_links.csv`; resumable; `--limit` as a seeded random sample that
   prints calls/s.
2. Fetch by `pmid -> doi -> pmcid`, record the key and source per row, and write `""` for a
   confirmed miss (never leave `null` behind).
3. Full-text mining only where `source.access.open_access` is true, from Europe PMC's OA XML;
   never store full text.
4. Measure precision on a sample before any load, and record the numbers in the README.
5. Only then: add an `identifiers` write mode (the five leaf paths + `schema_version`) to
   `moros_write.py` with tests, a `load_fields.py` mapper, and go through `moros-write`
   (dry run → `--limit` → confirm → verify).
6. If a field needs to become a list, that is a `schema.py` change: `schema-sync` first.
