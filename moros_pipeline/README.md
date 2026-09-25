# moros_pipeline

Everything that talks to **moros** — the MongoDB server hosting `dome_observatory.Content`, the
corpus behind the public DOME Observatory — plus the machinery that decides what to fetch, classify
and load next.

This is *the* refresh runbook. Before it existed, the corpus had been loaded exactly once, by hand,
through MongoDB Compass, with no record of how. The corpus is refreshed every two months.

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
verdict" a structural property rather than a claim about the prompt. `preprints` reaches exactly
the three Europe PMC identity fields, and `data_links` only the `data_links.*` leaves — not
`identifiers.*`, which a later, separate mode derives from them. From schema v1.6.0 the modes that
change a published value (`enrichment`, `licences`, `preprints`, `data_links`, `identifiers`) also
stamp `record_modified` on each document whose values actually change; `citations` never does, so a
count refresh never makes OAI-PMH harvesters re-fetch the corpus.

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
| `migrate_v1_4_0.py` | The in-place v1.2.0 → v1.4.0 shape bump (preprint fields + `data_links` at never-looked-up values), same constant-inverse pattern |
| `migrate_v1_5_0.py`, `migrate_v1_5_1.py` | Version-stamp migrations: v1.5.0's change lives in field values, v1.5.1's in the vocabularies |
| `migrate_v1_6_0.py` | v1.5.1 → v1.6.0: the version and one constant `record_modified` stamp on every document |
| `fetch_citations.py` | Europe PMC `citedByCount`, three keyed passes, resumable, refresh-aware |
| `join_citations.py` | Joins those counts onto document `_id`s |
| `load_fields.py` | Partial `$set` of allowlisted paths on documents that already exist |
| `load_documents.py` | Upserts whole new documents; `--reverse` deletes only ids it recorded inserting |
| `ensure_indexes.py` | Idempotent index check/creation; measures whether a citation index is warranted |
| `verify_corpus.py` | The invariants, the AlphaFold acceptance probe, and the manual post-load checklist |
| `build_release_metadata.py` | A release's DCAT / schema.org description, from the verify report, into dome-ml-observatory's `metadata/` ([docs/release_metadata.md](../docs/release_metadata.md)) |
| `export_journal_for_enrichment.py` | One journal's records out of moros, as enrichment input |
| `load_enrichment.py` | An enrichment event log back into moros, in place |
| `coverage_ledger.py` | The search space, and which (query, window) pairs are covered |
| `epmc_search.py` | Minimal cursorMark search client (a documented mirror of `ingest/epmc_client.py`) |
| `fetch_search_space.py` | Fetches only the windows not already covered |
| `build_incoming_documents.py` | Reduces a fetched window to records moros has never seen |
| `../../mongo_landscape_export/scripts/build_staged_documents.py` | Staged CSV + classification events -> documents JSONL |
| `load_fields.py --mode licences` | Backfills `source.access.license` + `open_access` on documents already loaded |
| `fetch_epmc_metadata.py` | Europe PMC identity, preprint server and data-links summary per record: batched `core` search, preprints keyed by DOI under `SRC:PPR` |
| `fetch_annotations.py` | Text-mined accession numbers per record, 8 ids per call to the annotations API — the primary link source |
| `fetch_datalinks.py` | Scholix `/datalinks` per record, for the residual text mining cannot cover (cross-references, data citations) |
| `import_textmined_bulk.py` | Europe PMC's monthly text-mined FTP dump, joined locally: cross-check and fallback, zero API calls |
| `datalinks_resources.py` | The resource catalogue: scheme / publisher / DOI prefix → a stable slug, label and category |
| `build_data_links.py` | Merges every route per document (dedupe, caps, BioStudies derivation) → `pid_preprints.csv`, `pid_data_links.csv` |
| `load_fields.py --mode preprints` / `--mode data_links` | Writes those two files through their allowlists |

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

# 4b. Data links for the batch. The staged CSV already carries epmc_source / epmc_id and the
#     data-links summary (build_incoming_documents.py read them off the core search), so no
#     metadata pass is needed -- only the link fetches, then the merge. Europe PMC for every
#     record; EBI Search for the batch's positives (schema v1.5.0), which needs the events.
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

# 5. Staged CSV + classification events -> documents, through the same schema.build_document()
#    the landscape and curated paths use.
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv \
    --data-links ../output/incoming_new_pid_data_links.csv \
    --identifiers ../output/incoming_new_pid_identifiers.csv

# 6. Load, index, verify.
python3 load_documents.py --input ../../mongo_landscape_export/output/incoming_new_documents.jsonl   # dry run: no --confirm
python3 load_documents.py --input ... --confirm
python3 ensure_indexes.py && python3 verify_corpus.py
```

Then the manual post-load checklist below — a service restart is the only way new facet values
appear.

## Enrichment — a separate, deliberate, costly cycle

Enrichment is **not** part of the recurring loop. It is billed at roughly nine times more per record than
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

python3 load_enrichment.py --events ../output/enrichment_flagship4_events.csv   # dry run: no --confirm
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
  records agree on all six fields even when the cheap pass flags none). Tokens × list price put
  enrichment at $4.07/1,000 records on V4-Flash while the bill was about $10/1,000; on V4.1 Flash it is
  billed at about **$1.80/1,000**, measured, and that is the price.
- **`enrich` needs a larger token budget than `classify`.** At the inherited `max_tokens=6000`, 12
  of 25 Bioinformatics records came back as `parse_error` with `output_tokens` of *exactly* 6000 —
  truncated mid-JSON. Raising it to 16,000 took that to 0 of 25. Billing is per token generated,
  not per the ceiling.

## Europe PMC data links and identity — the retrospective pass

Four routes, each used for what it is best at, and every one run hot: `--max-workers` defaults to
64, per-request timeouts are short, retries happen only on 429/5xx, and every `--limit` sample
prints achieved calls/s and the error rate so the worker count for the full run is a measurement,
not a guess. Raise workers between runs until errors appear.

| Route | Endpoint | Batching | Targets | Calls for the corpus |
|---|---|---|---|---|
| Identity + summary | `/search`, `resultType=core` | 300 pmids / 120 DOIs / 200 pmcids per call | every document | ~3,400 |
| Text-mined accessions | `annotations_api/annotationsByArticleIds` | 8 ids per call (hard limit) | `hasTMAccessionNumbers` Y (~50%) | ~53,000 |
| Scholix residual | `/{source}/{id}/datalinks` | none (one GET per record) | db cross-refs or `related_data` (~1.6%) | ~13,300 |
| FTP dump (optional) | `ftp.ebi.ac.uk/pub/databases/pmc/TextMinedTerms/` | whole files | local join | 0 |

```bash
cd scripts
python3 ../../scripts/export_corpus_keys.py
python3 fetch_epmc_metadata.py --limit 3000 && python3 fetch_epmc_metadata.py
python3 fetch_annotations.py  --limit 3000 && python3 fetch_annotations.py --max-workers 128
python3 fetch_datalinks.py    --limit 3000 && python3 fetch_datalinks.py
python3 build_data_links.py --report-only     # unmapped schemes / DOI prefixes -> datalinks_resources.py
python3 build_data_links.py
python3 migrate_v1_4_0.py && python3 migrate_v1_4_0.py --confirm   # after the v1.4.0 release is cut
python3 load_fields.py --mode preprints   && python3 load_fields.py --mode preprints --limit 500 --confirm && python3 load_fields.py --mode preprints --confirm
python3 load_fields.py --mode data_links  && python3 load_fields.py --mode data_links --limit 500 --confirm && python3 load_fields.py --mode data_links --confirm
python3 verify_corpus.py
```

Every fetch is resumable and `--max-age-days` refreshes only what has aged; data citations accrue,
so a data-links refresh every six months is reasonable. `build_data_links.py` never writes a
record's link detail until every route it was targeted for has answered, so `data_links.fetched_at`
is never a claim of completeness for a half-fetched record; `--datalinks-scope none` builds without
the Scholix route when that endpoint is down.

Measured on 2026-09-14:

- **The corpus metadata pass took 372 s.** 3,079 batched `core` calls at 96 workers: 8.3 calls/s,
  ~2,200 records/s, zero failed calls. A 300-id `core` call takes ~11 s server-side, so
  concurrency is the lever, not batch size. 486,160 records have `hasData`, 394,400 have text-mined
  accessions.
- **The corpus annotations pass took 186 s.** 48,668 calls of 8 ids at 128 workers: 262 calls/s,
  zero failed calls; 389,338 records, 2,269,089 text-mined accessions, a 509 MB JSONL.
- **A DOI-keyed chunk mostly cannot be attributed.** 90% of the plain `doi` pass came back as
  PMC-source records carrying no `doi` field; the pmid, pmcid and preprint passes were ~100%.
  The fetch asks every such key again: through the pmcid our corpus row holds, 200 per call
  (10,471 keys in 53 calls), and one by one for the rest (26); 23 s in all. A single quoted-DOI
  `core` query ran at ~3 calls/s at 64 workers, which is why the batched route goes first. Identity
  coverage afterwards: every DOI and PMCID key answered, 10 PMIDs and 3 preprint DOIs that Europe
  PMC has no record for.
- **Europe PMC has more source codes than MED/PPR/PMC/AGR/PAT.** The corpus holds MED 764,679,
  PPR 56,343, PMC 13,906, AGR 1,104, ETH 93 (theses) and CTX 21; schema v1.4.0 lists all of Europe
  PMC's codes.
- **Europe PMC's preprint server names agree with the verified DOI-prefix table** for 55,757 of the
  56,337 preprints it answered. The rest are its own names for the F1000 gateways (`F1000Res`,
  `Open Res Europe`, `Wellcome Open Res`, ...) and abbreviations (`NIHR Open Res`), which the
  API-wins rule keeps, so those cards will read the abbreviated name.
- **`hasData` and `dataLinksTagsList` exist only in `resultType=core`**, not `lite`. The corpus
  search space has 514,655 of 889,637 records with `HAS_DATA:y` (58%); in a 1,182-record corpus
  sample 50% had text-mined accessions, 1.3% curated cross-references, 35% supplementary files.
- **The annotations API takes at most 8 article ids per call** (a ninth returns HTTP 400 with
  "must contain between 1 and 8 values") and answered in 0.12–0.30 s.
- **`/datalinks` returned HTTP 500 for every id tried**, including Europe PMC's own documented
  example, after ~30 s each; `labsLinks` at 32 concurrent workers saw 27% timeouts. It was still
  failing at the end of the corpus run (20 of 20 after retries), so the corpus build used
  `--datalinks-scope none`. That is why the Scholix route is the residual and has short timeouts,
  and why the build can run without it.
- **`PMID:` is not a search field** (0 hits); `EXT_ID:` is, as `build_clause()` already does.
- **A BioStudies supplementary entry needs no call**: a PMC article with supplementary files is
  `S-EPMC<pmcid digits>`.
- **The FTP dump is uneven**: in the 2026-08-31 snapshot eight files had content (doi 423 MB, gen
  137 MB, nct, pdb, refseq, refsnp, rrid, uniprot) and most others were 0 bytes, so it is a
  cross-check, not the primary route.
- **A DOI in a reference list is a citation, not data.** Text-mined DOIs are kept only outside the
  References section and only for a data-repository prefix (`datalinks_resources.DOI_PREFIXES`).
- **`bookOrReportDetails.publisher` is a preprint server only on a PPR record.** Europe PMC fills it
  for 234 MEDLINE health-technology reports (NIHR Journals Library, CADTH), 93 EThOS theses (the
  university) and 17 CTX records as well; the pipeline keeps it only where the source is PPR.
- **BioStudies' supplementary-file mining links some accessions to the resource's own site**, not
  identifiers.org, and gives them no type: GISAID (`gisaid.org/EPI_ISL/`, 33,414 in the first
  200,000 records), OMIM (`omim.org/entry/`, 9,556), Human Protein Atlas, PDBe.
  `datalinks_resources.HOST_SCHEMES` types them by host; an untyped accession is otherwise dropped.
- **The corpus merge took 169 s at a 591 MB peak** in 8 shards, over 509 MB of annotations.
  Result: 846,703 of 846,716 documents with a Europe PMC identity, 56,866 preprints with a server,
  307,148 documents with at least one linked resource, 1,106,450 links across 73 resources
  (BioStudies 270,053 documents, ClinicalTrials.gov 25,137, GEO 20,379, ENA 17,830, PDB 16,743,
  dbSNP 7,482, UniProt 7,005, Zenodo 6,320, ...), and 1,014,393 literature DOIs left out.
- **Text-mined identifiers are not identifiers until cleaned, and a DOI is not a DOI until doi.org
  says so.** Stored verbatim, the first load gave 1,513 documents links that 404
  (`10.5281/zenodo.18675888.`, `10.6084/m9.figshare.24123303”.`, comma lists, `×` for `x`,
  `zenodo.XXXXX`, free-text RRIDs). The rebuild through `link_identifiers.py` repaired 8,955
  identifiers (7,914 RRIDs, 7,869 of all repairs taken from Europe PMC's own resolver URL; 438
  Zenodo, 196 OSF, 147 figshare DOIs), confirmed 16,627 DOI candidates at `doi.org/api/handles`
  in 100 s at 64 workers (166/s, no failures), dropped 395 DOIs doi.org does not have and 181
  accessions nothing clean could be recovered from, and changed 3,504 documents. All 50 sampled
  repaired DOIs resolved. Europe PMC's resolver URL is the better source for RRID and Orphanet
  (clean in 1,809 of 1,883 dirty cases) but repeats the junk for DOIs (5,399 of 5,848), which is
  why DOIs go to doi.org instead.

## EBI Search data links — positives, schema v1.5.0

EBI Search indexes the entries in EMBL-EBI's databases, and in several it mirrors, that name a
paper. Where the entry is an asset from the paper (a deposit, the paper's bio.tools record, its
DOME Registry report, its BioStudies supplementary files), `build_data_links.py` merges it into the
paper's `data_links` beside the Europe PMC routes, for positives only, and fills
`identifiers.dome_registry`. [`docs/data_links_sources.md`](../docs/data_links_sources.md) lists
the accepted and refused domains, the dedupe rules and the provenance fields; the accept list is
`scripts/ebisearch_resources.py`. Evaluated and decided 2026-09-14.

| Route | Endpoint | Batching | Targets | Measured 2026-09-14 |
|---|---|---|---|---|
| Discovery | `ebisearch/ws/rest/europepmc/entry/{pmid}/xref` | none | every positive PMID | 316,524 calls in 34 min at 256 workers (154/s), 49 connect timeouts |
| Detail | `.../europepmc/entry/{pmid,...}/xref/{domain}?size=100` | 100 PMIDs per call | discovered pairs in domains not dumped | 965 calls in 266 s, 95,351 pairs, no failures |
| Whole-domain dump | `ebisearch/ws/rest/{domain}?query=domain_source:{domain}` | 100 entries per page | the 29 accepted domains of at most 100,000 entries | 356,888 entries in 3,585 calls, 523 s (evaluation: 63 citing domains, 924,858 entries in 9,282 calls) |

```bash
cd scripts
python3 fetch_ebisearch_domains.py --list          # the citing domains
python3 fetch_ebisearch_domains.py                 # the accepted ones, dumped whole (--all-citing: every one)
python3 fetch_ebisearch_xrefs.py discover --limit 3000 && python3 fetch_ebisearch_xrefs.py discover --max-workers 256
python3 fetch_ebisearch_xrefs.py discover --record-failures   # a re-run: record what still fails
python3 fetch_ebisearch_xrefs.py detail            # accepted domains over the cap (--all-domains: every one)
python3 build_data_links.py --report-only          # EBI Search counts, rejected domains, unmatched values
python3 build_data_links.py                        # -> pid_data_links.csv, pid_identifiers.csv
python3 compare_data_links.py --old ../output/pid_data_links.v140.csv --new ../output/pid_data_links.csv
```

- **`size=100` is not optional.** Without it an xref answer carries one reference per entry
  whatever its `referenceCount` (pdbekb for 33024307: 1 of 5). The detail fetch checks every entry
  against the count and asks a short one again.
- **EBI Search keys `europepmc` by PMID only**; PMC and PPR ids answer no domains. The positives
  without a PMID (41,205, 88% preprints in a 1,000 sample) are reached only through the dumps:
  DOI, PMCID and PPR queries against the eight large domains matched none of that sample.
- **Paging stops at 100,000**: `start=100000` answers 200 with no entries, hence the dump cap.
- **Discovery ran 79.7 calls/s at 128 workers and 154/s at 256**, with 0.015% connect timeouts; a
  re-run recovered all but PMID 18575676, which answers 5xx every time.
- **Index quirks the dump tolerates, each the same on every read**: physiome holds an entry with
  no id; ega (EGAS00001006372) and biostudies-arrayexpress (E-MTAB-17162) index one entry twice.
  `_manifest.json` records them per domain.
