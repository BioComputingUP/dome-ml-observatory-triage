# moros_pipeline

Everything that talks to **moros** — the MongoDB server hosting `dome_observatory.Content`, the
corpus behind the public DOME Observatory — plus the machinery that decides what to fetch, classify
and load next.

This is *the* refresh runbook. Before it existed, the corpus had been loaded exactly once, by hand,
through MongoDB Compass, with no record of how. The corpus turns over 6–12 times a year.

**Standalone by design.** Like `epmc_licensing/` and `mongo_landscape_export/`, this folder sits
outside the `dome_triage` package and does **not** run through Docker: plain `python3` on the host,
no `dome_triage` import. It depends on `mongo_landscape_export/scripts/` (for `pid.py`, `schema.py`
and `citations_index.py`) and nothing depends on it — one direction only. The two folders are the
halves that would be extracted together into their own repository.

## Setup

```bash
pip install --user pymongo          # the one dependency beyond pandas / tqdm / requests / pyyaml
cp .env.example .env                # then fill in the real MONGODB_URI
python3 scripts/verify_corpus.py    # read-only: proves the connection and prints the invariants
```

`.env` is gitignored. The server has no authentication, so nothing in it is a credential — but the
host address is internal, and the sibling repo was sanitised for exactly that reason.

## How writing to moros is kept safe

Every write goes through `moros_write.py`, which enforces four properties in code rather than in a
procedure someone has to remember:

| | |
|---|---|
| **Dry run is the default** | `--confirm` is required to write anything; `--limit N` does a real, reversible N-document trial first |
| **An allowlist of leaf field paths per mode** | A path outside the mode's set raises *before* a single write is issued. Every entry is a leaf: `$set` on a group path such as `source` would replace the whole subdocument and drop its siblings |
| **A rollback snapshot before the first batch** | The prior values of exactly the paths about to change, for exactly the `_id`s about to change, flushed to `output/rollback/<run_id>.jsonl` before anything is written. Absent-vs-null is recorded separately, so a rollback restores "this field did not exist" rather than an explicit null |
| **No delete, no drop, no whole-document replace** | Only `$set` of allowlisted leaves. `load_documents.py` can replace a whole document, but refuses when the existing one differs unless told otherwise |

The allowlists are the interesting part. `citations` cannot reach `decision_provenance` — refreshing
a number must not be able to relabel who decided a record. `enrichment` cannot reach
`llm_classification` at all, which is what makes "enrichment is additive and cannot revise a
verdict" a structural property rather than a claim about the prompt.

Undo any field write:

```bash
python3 scripts/moros_write.py --rollback output/rollback/<run_id>.jsonl --confirm
```

**Never drop and reimport.** Dropping `Content` destroys `positives_text` — ~3 minutes of tokenising
plus a 1.5GB read to rebuild — and while it is gone `observatory-ws` silently degrades to a regex
scan with no error logged anywhere. "Search still works" is not evidence the index is there;
`ensure_indexes.py` is.

## The scripts

| Script | What it does |
|---|---|
| `moros_client.py` | The only module that opens a connection. Read-only by shape |
| `moros_write.py` | The safe writer, plus `--rollback` replay |
| `migrate_v1_2_0.py` | The in-place v1.1.0 → v1.2.0 shape bump, with a documented constant inverse |
| `fetch_citations.py` | Europe PMC `citedByCount`, three keyed passes, resumable, refresh-aware |
| `join_citations.py` | Joins those counts onto document `_id`s |
| `load_fields.py` | Partial `$set` of allowlisted paths on documents that already exist |
| `load_documents.py` | Upserts whole new documents; `--reverse` deletes only ids it recorded inserting |
| `ensure_indexes.py` | Idempotent index check/creation; measures whether a citation index is warranted |
| `verify_corpus.py` | The invariants, the AlphaFold acceptance probe, and the manual post-load checklist |
| `export_journal_for_enrichment.py` | One journal's records out of moros, as enrichment input |
| `load_enrichment.py` | An enrichment event log back into moros, in place |
| `coverage_ledger.py` | The search space, and which (query, window) pairs are covered |
| `epmc_search.py` | Minimal cursorMark search client (a documented mirror of `ingest/epmc_client.py`) |
| `fetch_search_space.py` | Fetches only the windows not already covered |
| `build_incoming_documents.py` | Reduces a fetched window to records moros has never seen |
| `../../mongo_landscape_export/scripts/build_staged_documents.py` | Staged CSV + classification events -> documents JSONL |
| `load_fields.py --mode licences` | Backfills `source.access.license` + `open_access` on documents already loaded |

Tests: `cd scripts && python3 -m pytest .` — hermetic, no server, no network.

## Recurring batch run — new Europe PMC records, end to end

Every step is independently runnable and resumable; nothing here has to be done in one sitting.

```bash
cd scripts

# 1. Fetch only the time windows this exact query has never covered. Cross-checks moros and
#    STOPS if moros is ahead of the ledger rather than guessing which windows are safe to skip.
python3 fetch_search_space.py --dry-run
python3 fetch_search_space.py --up-to today

# 2. Reduce to records moros has never seen. Mints each record's deterministic UUID5 and drops
#    every _id already in the corpus, so only genuinely-new papers survive.
python3 build_incoming_documents.py --incoming ../output/incoming/<query-hash>

# 3. Classify them. Paid, and gated: project the cost first, then pass it back explicitly.
#    The event log defaults to <input>_classification_events.csv beside the staged file, so
#    batches are never conflated.
docker compose run --rm pipeline dome-triage llm-classify classify \
    --scope staged_file --input /app/moros_pipeline/output/incoming_new.csv \
    --tier flash --estimated-usd <X> --confirm

# 4. Citations AND licences, in the metadata stage, before loading.
#    --with-licence switches resultType to core, which is required rather than preferred: lite
#    carries no `license` field at all. New records need both exactly once.
python3 fetch_citations.py --with-licence --input ../output/incoming_new.csv
python3 join_citations.py --citations ../output/epmc_citations.csv \
    --with-licence --corpus-from-moros --output ../output/pid_licences.csv
#    A later routine refresh wants citations ONLY -- licences do not change, and `lite` is
#    lighter: python3 fetch_citations.py --max-age-days 30

# 5. Staged CSV + classification events -> documents, through the same schema.build_document()
#    the landscape and curated paths use.
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv

# 6. Load, index, verify.
python3 load_documents.py --input ../../mongo_landscape_export/output/incoming_new_documents.jsonl --dry-run
python3 load_documents.py --input ... --confirm
python3 ensure_indexes.py && python3 verify_corpus.py
```

Then the manual post-load checklist below — a service restart is the only way new facet values
appear.

## Enrichment — a separate, deliberate, costly cycle

Enrichment is **not** part of the recurring loop. It costs roughly 13,000x more per record than
classification and is run per journal or per cohort, when wanted.

```bash
# One or many journals in a single export and a single events file. --journal is repeatable, and
# names are matched VERBATIM: Science is stored as "Science (New York, N.Y.)".
# --max-usd (default 20) refuses to write an input whose projection exceeds it -- `enrich` itself
# has no budget gate, so this is where a runaway is stopped, before any money is spent.
python3 export_journal_for_enrichment.py \
    --journal "Bioinformatics (Oxford, England)" \
    --journal "Nature" --journal "Science (New York, N.Y.)" --journal "Cell" \
    --out ../output/enrich_input_flagship4.csv

# ALWAYS pass --events-out: the record_id here is the Mongo _id (a UUID5), while the Step 20j
# trial log's is a sha1. Appending one to the other conflates two identifier spaces in one log.
docker compose run --rm pipeline dome-triage llm-classify enrich \
    --tier flash --concurrency 60 \
    --input /app/moros_pipeline/output/enrich_input_flagship4.csv \
    --events-out /app/moros_pipeline/output/enrichment_flagship4_events.csv

python3 load_enrichment.py --events ../output/enrichment_flagship4_events.csv --dry-run
python3 load_enrichment.py --events ... --limit 25 --confirm    # real, reversible trial
python3 load_enrichment.py --events ... --confirm
```

Already-enriched documents are excluded from the export, and `enrich` resumes from its own event
log, so a re-run never re-pays for finished work. About 3% of records hit the 16,000-token cap and
are recorded with `finish_reason: length`; they are **not** retried automatically, because retrying
at the same cap re-truncates deterministically — raise `ENRICHMENT_MAX_TOKENS` first, then pass
`--retry-truncated`.

## The search space is a config file, not a constant

`config/search_space.yaml` holds the query. It used to be `AI_ML_QUERY` hardcoded in
`src/dome_triage/ingest/bulk_match.py`, so changing the corpus meant changing code.

Coverage is keyed on `sha256` of the canonical query, so **editing the terms invalidates coverage
for the changed query only** — the new query starts from nothing rather than inheriting the old
one's windows, while a cosmetic reordering changes nothing. `search_space_expansion/` has live
measured counts for candidate terms worth adding (BERT 38,074 · GPT 30,185 · U-Net 13,094 · …) and
the ones to avoid (`neural network` 1.55M, almost all biological).

Two authorities, deliberately not merged: the **ledger** knows what was *fetched* (including
windows that legitimately returned nothing); **moros** knows what was *loaded*. The ledger may be
ahead. moros being ahead means the ledger lost history, and the run stops — guessing there would
silently re-create the exact gap this pipeline exists to close.

## After any load — manual, and nothing does it for you

1. **Restart `observatory-ws`.** `FacetsService` is boot-loaded with **no TTL**, so new journals,
   MeSH terms and licences never appear until it restarts. `StatsService`, `CountService` and
   `JournalsService` are 24h TTL.
2. Re-run `python3 schema/generate_facet_stats.py --from-api <url>` in `../dome-ml-observatory` and
   check `/api/stats` reconciles.
3. Cut a schema release there with the `schema-version` skill if the shape changed. Never hand-edit
   a published `schema/releases/vX.Y.Z/` folder.

## Things measured here, so nobody has to rediscover them

- **`mongoimport` cannot complete a load against this server.** It stalls around 17% of a 21.4MB
  file, reports `use of closed network connection`, and exits non-zero having written exactly one
  1,000-document batch — three attempts, same result. It also reported `0 document(s) imported
  successfully` for a batch that *had* landed. The same host took 811,036 pymongo bulk writes with
  zero errors, so `load_documents.py` uses `ReplaceOne(upsert=True)` and treats the collection count
  as the only honest authority. Both roadmaps name `mongoimport`; this is why the code does not.
- **Europe PMC `citedByCount` is in the cheap `resultType=lite`** (the licence fetch needs `core`).
  Per-pass URI byte ceilings, binary-searched live: pmid 300/chunk, doi 120 (150 works, 200 → HTTP
  414), pmcid 200. Measured yield: pmid 99.6%, pmcid 100%, doi 80.7% — 98.1% overall.
- **A lexical prefix is not a sample.** The first 480 no-pmid DOIs are a single cluster of Cochrane
  reviews EPMC does not index by DOI, giving a 10.2% hit rate where the real population yields
  80.0%. `--limit` therefore takes a seeded random sample.
- **HTML entities in `title` and `abstract` are decoded to stability, not once.** 1.85% of titles
  and 0.36% of abstracts carry entities; ~135 abstracts are encoded *twice*, and those are `<`/`>`
  used as comparison operators ("ranging from <50 to >25,000"). `authors`, `journal` and
  `rationale` measured 0.000% affected and are deliberately left alone.
- **A detached `docker compose run` exits non-zero while its container keeps working — never
  retry on that exit code.** `docker-compose.yml` sets `tty: true` on the `pipeline` service for
  interactive use. Detached (`nohup`/`setsid`/background), the compose *client* returns 1 and
  produces no output, but **the container is still running and still doing the work**, writing to
  the mounted volume as normal. `-T` suppresses the TTY allocation but does not change this.

  This is a trap with teeth: a retry loop that treats exit 1 as failure starts another container
  on top of the one still running. That happened here — **ten concurrent containers**, all
  enriching overlapping records, producing 3,935 event rows for 1,424 records and **$9.63 of
  duplicated paid work**.

  The rule: for any long detached run, **judge progress by the output file, and start a new
  container only when `docker ps` confirms none is running.** `/tmp/enrich_supervisor.sh`'s shape
  is the pattern — poll unique record count, check `docker ps`, never trust the exit code.
- **Don't rebuild the image or start a second `docker compose run` while a job is in flight.**
  Resumability makes the disruption recoverable, not free.
- **Licence is fetched by three keys, not just pmid.** The original
  `epmc_licensing/fetch_licensing.py` builds `EXT_ID:` clauses under `SRC:MED`, so the 67,985
  corpus records with no pmid were structurally unfetchable from the day it ran -- recorded as
  "unmatched", which reads like an EPMC coverage limit rather than a key-choice one. 74,472
  documents sat at `license: null` because of it. `fetch_citations.py --with-licence` reuses the
  `pmid -> doi -> pmcid` passes and closes that.
- **`""` and `null` are different licence answers and must stay different.** `null` is "never
  looked up"; `""` is "looked up, EPMC disclosed none". A key EPMC cannot answer therefore has to
  be written as `""`, not dropped -- dropping it leaves the document `null` and every future
  backfill fetches it again, forever. 12,980 DOI-keyed lookups hit this: EPMC returns the paper as
  a PMC-source record carrying neither the DOI nor a licence, so it cannot be attributed to one of
  the 120 keys in the chunk, but "no licence disclosed for this key" is still true.
- **`reasoning_effort` is not a cost lever, and thinking cannot be turned down.** Measured over
  four paired 100-record arms (`thinking_ablation/` Round 3): `low` cost 2% *more* than the default
  with six times the violations, and every thinking-enabled arm spends 98% of its output on
  reasoning whatever level is requested. A hierarchical domain rendering saved nothing either, and
  a cheap-first two-pass has a useless trigger — violations do not predict disagreement (0% of
  records agree on all six fields even when the cheap pass flags none). Enrichment costs
  **$4.07/1,000 records** and that is the price.
- **`enrich` needs a larger token budget than `classify`.** At the inherited `max_tokens=6000`, 12
  of 25 Bioinformatics records came back as `parse_error` with `output_tokens` of *exactly* 6000 —
  truncated mid-JSON. Raising it to 16,000 took that to 0 of 25. Billing is per token generated,
  not per the ceiling.
