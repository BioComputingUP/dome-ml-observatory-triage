---
name: citations-refresh
description: >
  Refresh Europe PMC citation counts on the records already in moros: export the corpus keys
  read-only, re-fetch only counts older than N days, join them to document ids, and write them
  through the `citations` allowlist via moros-write. Trigger on "refresh citation counts",
  "update citations", "pull the new citation counts", every two months. Free (Europe PMC), and separate
  from the batch load, which fetches counts for new records itself.
---

# citations-refresh

Every document's `publication_metadata.citation_count` carries `citation_count_updated`
(ISO-8601) and `citation_source`. The fetch is resumable and age-aware: it reads its own previous
output (`moros_pipeline/output/epmc_citations.csv`) and re-fetches only keys whose last fetch is
older than `--max-age-days`. The `citations` write mode can reach exactly those three fields and
nothing else (not `decision_provenance`, not licences).

Do **not** pass `--with-licence` on a routine refresh: licences do not change and `core` is a
much heavier response. Licences are fetched once, when a batch is loaded (`moros-write` A).

## Procedure (host, VPN, `moros_pipeline/.env`)

```bash
cd moros_pipeline/scripts

# 1. The key list must come from the live collection, not a file: every document added since the
#    first load exists only in moros. Read-only, ~1-2 minutes for the whole corpus.
python3 ../../scripts/export_corpus_keys.py                 # -> ../output/corpus_keys.csv

# 2. Smoke, then the refresh. Keys fetched within the last N days are skipped.
python3 fetch_citations.py --input ../output/corpus_keys.csv --max-age-days 30 --limit 2000
python3 fetch_citations.py --input ../output/corpus_keys.csv --max-age-days 30
#    Genuine EPMC misses (Cochrane reviews, some preprints; ~1.9% of the corpus) recur on every
#    run. That is expected and is reported as such, not a bug.

# 3. Join to document ids from moros (never from EPMC's response).
python3 join_citations.py --corpus-from-moros --report-only
python3 join_citations.py --corpus-from-moros                # -> ../output/pid_citations.csv

# 4. Write, through moros-write section C: dry run, trial, confirm, verify.
python3 load_fields.py --mode citations
python3 load_fields.py --mode citations --limit 500 --confirm
python3 load_fields.py --mode citations --confirm
python3 verify_corpus.py
```

Yield to expect per key (measured): pmid 99.6%, pmcid 100%, doi 80.7%; overall ~98.1% of the
corpus carries a count. Throughput ~840 records/s, so the whole corpus takes ~17 minutes when
everything is stale; a refresh with `--max-age-days 30` re-fetches only what has aged.

## Report

Keys targeted / skipped as fresh / fetched / missed; rows joined; documents matched / modified
by the load; the run id and rollback path; the new "older than 30 days" count from
`python3 scripts/cost_dashboard.py --live`. Remind the user that `citation_count` is not
indexed, so a deep "most cited" sort stays slow until `ensure_indexes.py
--measure-citation-sort` says an index is warranted.

## If the dashboard shows documents with `license: null`

That is "never looked up", and a key-choice gap, not an EPMC one. Close it with the licence
passes instead:

```bash
python3 fetch_citations.py --targets moros:missing-licence --with-licence --output ../output/epmc_licence_backfill.csv
python3 join_citations.py --citations ../output/epmc_licence_backfill.csv --with-licence --corpus-from-moros --output ../output/pid_licences.csv
python3 load_fields.py --mode licences   # then --limit / --confirm via moros-write C
```
