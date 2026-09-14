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

Always, in this order: `--dry-run` (or no `--confirm`) → `--limit N --confirm` (a real, reversible
trial) → `--confirm` → `verify_corpus.py` → the post-load checklist. Report the run id and the
rollback path after every confirmed write.

## A. Load a classified batch as new documents

```bash
cd moros_pipeline/scripts
# 1. Citations AND licences for the batch, by pmid -> doi -> pmcid. --with-licence is required for
#    new records (the lite result type carries no licence).
python3 fetch_citations.py --with-licence --input ../output/incoming_new.csv
python3 join_citations.py --citations ../output/epmc_citations.csv \
    --with-licence --corpus-from-moros --output ../output/pid_licences.csv

# 1b. Europe PMC data links for the batch. The staged CSV already carries epmc_source / epmc_id,
#     the preprint server and the data-links summary; only the link fetches and the merge remain.
python3 fetch_annotations.py --input ../output/incoming_new.csv --output ../output/incoming_new_annotations.jsonl
python3 fetch_datalinks.py   --input ../output/incoming_new.csv --output ../output/incoming_new_datalinks.jsonl
python3 build_data_links.py --keys ../output/incoming_new.csv --metadata ../output/incoming_new.csv \
    --annotations ../output/incoming_new_annotations.jsonl --datalinks ../output/incoming_new_datalinks.jsonl \
    --out-preprints ../output/incoming_new_pid_preprints.csv --out-data-links ../output/incoming_new_pid_data_links.csv
#     (/datalinks down? add --datalinks-scope none; the next data-links refresh completes them)

# 2. Documents, through the same schema.build_document() every record was built with.
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv \
    --data-links ../output/incoming_new_pid_data_links.csv --report-only
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv \
    --data-links ../output/incoming_new_pid_data_links.csv
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
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --dry-run
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --limit 25 --confirm
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --confirm
```

The `enrichment` allowlist covers `content_filters`' six vocabulary fields and the
`llm_enrichment` group only; it cannot reach `llm_classification`. `parse_error` rows are skipped.
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
```

`preprints` writes `identifiers.epmc_id`, `source.epmc_source` and, only when Europe PMC gave one,
`publication_metadata.preprint_server`. `data_links` writes the `data_links.*` leaves only; it
cannot reach `identifiers.*`. Both files come from `build_data_links.py` (the `data-links` skill).
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

`python3 ensure_indexes.py` reports; `--confirm` creates a missing `class_year_id` (~4 s) or
`positives_text` (~3 min, ~1 GB of transient sort files). `--measure-citation-sort` says whether a
citation index is warranted yet. Do not create indexes speculatively on a shared production host.

## F. Shape migrations

```bash
python3 migrate_v1_4_0.py            # pre-flight: refuses any schema_version it does not expect
python3 migrate_v1_4_0.py --confirm  # one updateMany: 1.2.0 -> 1.4.0, new fields at never-looked-up values
python3 migrate_v1_4_0.py --reverse --confirm   # only before any preprints / data_links load
```

Run only after `check_alignment.py` shows v1.4.0 published. Rollback is a constant inverse, not a
snapshot, and `--reverse` refuses once real data has landed.

## After any confirmed write: the manual checklist

1. Restart `observatory-ws` (its facet cache is boot-loaded with no TTL).
2. In `dome-ml-observatory`: `python3 schema/generate_facet_stats.py --from-api <url>` and check
   `/api/stats` reconciles with `verify_corpus.py`.
3. If `SCHEMA_VERSION` or a vocabulary changed: `schema-sync` here, then their `schema-version`
   skill.
4. Keep the events file until this checklist is done; it is the paid record of the run.
