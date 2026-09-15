---
name: moros-write
description: >
  The only way anything is written to moros (dome_observatory.Content): loading a classified batch
  as new documents, merging an enrichment events file, refreshing citation or licence fields,
  rolling a write back, and checking or creating the indexes. Trigger on "load the batch",
  "load to moros", "merge the enrichment", "roll back run X", "check the indexes", "verify the
  corpus", or any request that changes the database. Dry run first, reversible trial second,
  confirm third, verify last, restart observatory-ws after.
---

# moros-write

Host Python from `moros_pipeline/scripts/`, on the VPN, with `moros_pipeline/.env`. Every writer
here is dry-run by default, restricted to an allowlist of leaf field paths per mode
(`moros_write.py::WRITE_MODES`), and snapshots the prior values of exactly the paths about to change
to `../output/rollback/<run_id>.jsonl` before the first batch. There is no drop, delete or
whole-document replace path. **Never drop `Content`**: it destroys `positives_text`, and search
silently degrades to a regex scan with no error anywhere.

Always, in this order: a dry run (the command without `--confirm`; no loader has a `--dry-run`
flag) → `--limit N --confirm` (a real, reversible trial) → `--confirm` → `verify_corpus.py` → the post-load checklist. Report the run id and the
rollback path after every confirmed write.

## A. Load a classified batch as new documents

```bash
cd moros_pipeline/scripts
# 1. Citations AND licences for the batch, by pmid -> doi -> pmcid. --with-licence is required for
#    new records (the lite result type carries no licence).
#    Into a batch file: the shared epmc_citations.csv predates the licence columns, so licences
#    appended there are invisible to the builder (2026-09-15).
python3 fetch_citations.py --with-licence --input ../output/incoming_new.csv \
    --output ../output/incoming_new_citations.csv

# 1b. Data links for the batch. The staged CSV already carries epmc_source / epmc_id, the preprint
#     server and the data-links summary; only the link fetches and the merge remain. Europe PMC for
#     every record; EBI Search for the batch's positives (schema v1.5.0), which needs the events.
python3 fetch_annotations.py --input ../output/incoming_new.csv --output ../output/incoming_new_annotations.jsonl
python3 fetch_datalinks.py   --input ../output/incoming_new.csv --output ../output/incoming_new_datalinks.jsonl
python3 fetch_ebisearch_domains.py --max-age-days 30        # EBI Search: re-dumps only stale accepted domains
python3 fetch_ebisearch_xrefs.py discover --input ../output/incoming_new.csv \
    --classification-events ../output/incoming_new_classification_events.csv   # the batch's positives
python3 fetch_ebisearch_xrefs.py discover --input ../output/incoming_new.csv \
    --classification-events ../output/incoming_new_classification_events.csv --record-failures
python3 fetch_ebisearch_xrefs.py detail
python3 build_data_links.py --keys ../output/incoming_new.csv --metadata ../output/incoming_new.csv \
    --annotations ../output/incoming_new_annotations.jsonl --datalinks ../output/incoming_new_datalinks.jsonl \
    --classification-events ../output/incoming_new_classification_events.csv --shards 1 \
    --out-preprints ../output/incoming_new_pid_preprints.csv --out-data-links ../output/incoming_new_pid_data_links.csv \
    --out-identifiers ../output/incoming_new_pid_identifiers.csv
#     (/datalinks down, or fetch_datalinks.py found no targets and wrote no file? add
#      --datalinks-scope none; the next data-links refresh completes them)

# 2. Documents, through the same schema.build_document() every record was built with.
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv \
    --citations ../output/incoming_new_citations.csv --licence-fetch ../output/incoming_new_citations.csv \
    --data-links ../output/incoming_new_pid_data_links.csv \
    --identifiers ../output/incoming_new_pid_identifiers.csv --report-only
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv \
    --citations ../output/incoming_new_citations.csv --licence-fetch ../output/incoming_new_citations.csv \
    --data-links ../output/incoming_new_pid_data_links.csv \
    --identifiers ../output/incoming_new_pid_identifiers.csv
#    -> ../../mongo_landscape_export/output/incoming_new_documents.jsonl + .report.json

# 3. Load. Pre-flight splits inserts from updates; for a new batch expect "all new".
python3 load_documents.py --input ../../mongo_landscape_export/output/incoming_new_documents.jsonl
python3 load_documents.py --input ... --limit 100 --confirm
python3 load_documents.py --input ... --confirm

# 4. Indexes and invariants.
python3 ensure_indexes.py                      # report; --confirm only if one is missing
python3 verify_corpus.py --expect-count <baseline + documents loaded>
```

Report: built / left out (no abstract, parse errors) / inserted / already existing / errors, the
new count, and whether every invariant passed. `load_documents.py` refuses to replace an existing
document whose content differs; do not pass `--allow-replace-existing` unless the user has decided
the existing document should lose what the new one lacks.

## B. Merge an enrichment events file

```bash
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv              # dry run
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --limit 25 --confirm
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --confirm
```

The `enrichment` allowlist covers `content_filters`' six vocabulary fields and the
`llm_enrichment` group only; it cannot reach `llm_classification`. `parse_error` rows are skipped.
Never merge an events file written for a comparison (`enrichment_revalidation_*`): it re-enriched
records that already carry an enrichment, and merging would overwrite what it was compared with.
Report matched / modified / violations, then `verify_corpus.py`.

## C. Field refreshes

```bash
python3 load_fields.py --mode citations --input ../output/pid_citations.csv            # dry run
python3 load_fields.py --mode citations --input ../output/pid_citations.csv --limit 500 --confirm
python3 load_fields.py --mode citations --input ../output/pid_citations.csv --confirm
python3 load_fields.py --mode licences  --input ../output/pid_licences.csv  ...         # same shape
```

`citations` cannot reach `decision_provenance`; `licences` writes `license` and `open_access`
only. A licence key EPMC cannot answer is written as `""` (looked up, none disclosed), never left
`null` (never looked up).

```bash
python3 load_fields.py --mode preprints  --input ../output/pid_preprints.csv   ...      # same shape
python3 load_fields.py --mode data_links --input ../output/pid_data_links.csv  ...      # same shape
python3 load_fields.py --mode identifiers --input ../output/pid_identifiers.csv ...     # same shape
```

`preprints` writes `identifiers.epmc_id`, `source.epmc_source` and, only when Europe PMC gave one,
`publication_metadata.preprint_server`. `data_links` writes the `data_links.*` leaves only; it
cannot reach `identifiers.*`. `identifiers` writes `identifiers.dome_registry` (the DOME Registry
entry, or `""` for a positive looked up with none); its allowlist is the five reserved identifier
leaves and `schema_version`, so it cannot reach `data_links.*`, the Europe PMC identity or any
verdict. All three files come from `build_data_links.py` (the `data-links` skill).
`load_fields.py --mode data_links` and `load_documents.py` refuse, before connecting, any input
whose link ids or URLs fail `link_identifiers.malformed_links()` ("refusing to load ... malformed
data link"). That means the staging file is stale or was edited: rebuild it with
`build_data_links.py`; never strip the offending rows by hand.

## D. Roll back

```bash
python3 moros_write.py --rollback ../output/rollback/<run_id>.jsonl --confirm      # any field write
python3 load_documents.py --reverse ../output/rollback/<run_id>.inserted.json --confirm   # deletes only ids that run inserted
```

Restores absent-vs-null faithfully. Run `verify_corpus.py` afterwards.

## E. Indexes

`python3 ensure_indexes.py` reports; `--confirm` creates a missing `class_year_id` (~4 s),
`positives_text` (~3 min, ~1 GB of transient sort files) or `record_modified_positive` (schema
v1.6.0: partial on positives, the keyset OAI-PMH and the sitemaps page by). `--measure-citation-sort` says whether a
citation index is warranted yet. Do not create indexes speculatively on a shared production host.

## F. Shape migrations

```bash
python3 migrate_v1_4_0.py            # pre-flight: refuses any schema_version it does not expect
python3 migrate_v1_4_0.py --confirm  # one updateMany: 1.2.0 -> 1.4.0, new fields at never-looked-up values
python3 migrate_v1_4_0.py --reverse --confirm   # only before any preprints / data_links load
```

Every migration runs only after `check_alignment.py` shows its release published, and before any
field load of that release (`schema/README.md`, "Release procedure"). Rollback is a constant inverse, not a
snapshot, and `--reverse` refuses once real data has landed.

```bash
python3 migrate_v1_5_0.py            # 1.4.0 -> 1.5.0: the version stamp only
python3 migrate_v1_5_0.py --confirm
python3 migrate_v1_5_0.py --reverse --confirm   # only before any v1.5.0 data_links / identifiers load
```

v1.5.0 adds keys inside the `data_links` arrays and fills `identifiers.dome_registry`; both arrive
through `load_fields.py`, so migrate first. `verify_corpus.py` fails on a document carrying v1.5.0
link keys under an older version, and `--reverse` refuses once EBI Search links or a DOME Registry
id have landed.

```bash
python3 migrate_v1_5_1.py            # 1.5.0 -> 1.5.1: the version stamp only
python3 migrate_v1_5_1.py --confirm
python3 migrate_v1_5_1.py --reverse --confirm   # always safe: v1.5.1 puts nothing in documents
```

v1.5.1 changes no field: the two modelling vocabularies gained `ontology_mappings`.

```bash
python3 migrate_v1_6_0.py                        # 1.5.1 -> 1.6.0: the version, plus one record_modified stamp on every document
python3 migrate_v1_6_0.py --confirm
python3 ensure_indexes.py --confirm              # then build record_modified_positive
python3 migrate_v1_6_0.py --reverse --confirm    # always safe: sets 1.5.1 back and removes record_modified
```

v1.6.0 adds `record_modified`, the datestamp OAI-PMH harvests by. From then on `SafeWriter` stamps it
itself in the `enrichment`, `licences`, `preprints`, `data_links` and `identifiers` modes, only on
documents whose values actually change, and `load_documents.py` on every document it writes. The
`citations` refresh never stamps, so a count refresh never forces a re-harvest.

## After any confirmed write: the manual checklist

1. Restart `observatory-ws` (its facet cache is boot-loaded with no TTL).
2. In `dome-ml-observatory`: `python3 schema/generate_facet_stats.py --from-api <url>` and check
   `/api/stats` reconciles with `verify_corpus.py`.
3. If `SCHEMA_VERSION` or a vocabulary changed: `schema-sync` here, then their `schema-version`
   skill.
4. Keep the events file until this checklist is done; it is the paid record of the run.
