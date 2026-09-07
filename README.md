# dome-observatory-triage

The data pipeline behind [DOME Observatory](https://observatory.dome-ml.org/): it fetches the
metadata of AI/ML publications from Europe PMC, classifies each one with a validated LLM
second-curator, enriches the positives with controlled vocabularies, and writes the result into
the corpus database (`dome_observatory.Content` on the host called **moros**) that the sister
repository serves read-only.

Two operational repositories, one database, one direction of writes:

| Repository | Role | Touches moros |
|---|---|---|
| **this one** — `dome-observatory-triage` | data management: fetch, dedupe, classify, enrich, format, load, refresh | **the only writer** |
| [`dome-ml-observatory`](https://github.com/BioComputingUP/dome-ml-observatory) | the production service: Angular UI + NestJS API, and the **published schema releases** | read-only |

Every command below is meant to be run one at a time by a person (or by an agent following the
skills in [`SKILLS.md`](SKILLS.md)), inspecting the output before the next step. Nothing here is a
black box, and nothing that costs money or writes to the database runs without an explicit
confirmation.

## The pipeline at a glance

```
Europe PMC ──fetch_search_space.py──▶ windows never fetched ──build_incoming_documents.py──▶ staged CSV
                                                                    (records moros has never seen)
                        │
                        ▼
      dome-triage llm-classify classify --scope staged_file        (DeepSeek V4 Flash, prompt v1)
                        │  classification events (positive / negative / undeterminable + rationale)
                        ▼
      fetch_citations.py --with-licence ──▶ citation counts + licences for the batch
                        │
                        ▼
      build_staged_documents.py ──▶ documents JSONL (schema 1.2.0) ──load_documents.py──▶ moros
                                                                  ensure_indexes.py · verify_corpus.py

positives already in moros ──export_journal_for_enrichment.py──▶ enrich (prompt e1) ──load_enrichment.py──▶ moros
moros ──fetch_citations.py --max-age-days N──▶ join_citations.py ──load_fields.py --mode citations──▶ moros
```

Two runtimes, deliberately:

- **Docker** runs the `dome-triage` Python package (the validated classification and enrichment
  engine; the CLI keeps that name). It never connects to the database.
- **Host Python 3** runs everything in `moros_pipeline/` and `mongo_landscape_export/`. Those are
  plain scripts with `pymongo`, `pandas`, `tqdm`, `requests`, `pyyaml`, and they are the only code
  that opens a connection to moros.

## Setup

```bash
# 1. Secrets. Both real files are gitignored.
cp .env.example .env                                  # DEEPSEEK_API_KEY
cp moros_pipeline/.env.example moros_pipeline/.env    # MONGODB_URI (internal host; lab VPN)

# 2. Host side (moros pipeline + document builders)
pip install --user pymongo pandas tqdm requests pyyaml
python3 moros_pipeline/scripts/verify_corpus.py       # read-only: proves the connection, prints invariants

# 3. Container side (classification + enrichment)
docker compose build
docker compose run --rm pipeline pytest --ignore=tests/test_curate_page.py    # the engine's test suite
```

`tests/test_curate_page.py` is excluded: it drives the inert Streamlit curation app through
`AppTest`, which resolves its script path relative to the test file and cannot find it from this
repository's root. The app is not part of the operational surface. Every test that is —
classification, enrichment, the DeepSeek client, the staged-file scope, the event-log lock, the
response parser, the Europe PMC client, provenance — runs and must pass.

No path in this repository names a machine. Config files refer to the external data
repositories as `${DOME_TRIAGE_DATA_ROOT}/...`, which defaults to this repository's parent
directory, so a checkout with the siblings beside it needs no configuration; set that one variable
if they live elsewhere. `python3 scripts/check_no_absolute_paths.py` fails if an absolute home path
ever reaches a tracked file.

Hermetic tests for the host-side scripts, no server or network needed:

```bash
(cd moros_pipeline/scripts && python3 -m pytest .)
(cd mongo_landscape_export/scripts && python3 -m pytest .)
```

## The processes, tersely

### 1. Triage — fetch new records and keep only the unseen ones

**What is fetched.** The search space is a config file, `moros_pipeline/config/search_space.yaml`:
`"artificial intelligence" OR "machine learning"`, every Europe PMC source (MED, PMC, PPR
preprints, AGR, PAT), windowed by first publication date from 1900. Each record comes back with
pmid / pmcid / doi, title, abstract, authors, journal, year, MeSH headings, publication types,
author keywords and the open-access flag. Licence strings and citation counts are fetched in a
separate keyed pass (step 3), because the cheap `lite` result type carries no licence.

**What "new" means.** `moros_pipeline/output/coverage_ledger.json` records every (query, year
window) ever fetched; coverage is keyed on the sha256 of the canonical query, so editing the terms
starts fresh coverage for the new query only. `build_incoming_documents.py` mints each record's
deterministic UUID5 (`mongo_landscape_export/scripts/pid.py`, `pmcid > doi > pmid`) and drops
every `_id` already in moros. The same paper always mints the same `_id`, whichever path it
arrives by, which is what makes every later load idempotent.

```bash
cd moros_pipeline/scripts
python3 fetch_search_space.py --show-query          # the exact query and its hash, no fetch
python3 fetch_search_space.py --dry-run             # which windows are missing
python3 fetch_search_space.py --up-to today         # fetch them (the current year is always re-fetched)
python3 build_incoming_documents.py --incoming ../output/incoming/<query-hash>
#   -> ../output/incoming_new.csv   (pid, pmid, pmcid, doi, title, abstract, journal, year, ...)
```

Free: Europe PMC charges nothing. The run stops, rather than guessing, if moros holds documents in
years the ledger has never recorded.

### 2. Classification — is this an AI/ML methods paper?

DeepSeek V4 Flash acts as an independent second curator. The system message is the preamble plus
the **full text of `curation_criteria/CRITERIA.md`** (prompt `v1`); the user message is exactly
`title / journal / year / abstract` and nothing else, so no label, MeSH term or prior decision can
reach the model. The answer is `positive`, `negative` or `undeterminable` with a two-sentence
rationale. Validated against human curation at kappa 0.81 before it was ever run at scale. Every
event records the criteria sha256 it was judged under; resumption keys on it, so a criteria edit
starts a fresh, non-conflated batch.

```bash
# Cost first (see COST_DASHBOARD.md; ~$0.0003 per record off-peak), then run:
docker compose run --rm pipeline dome-triage llm-classify classify \
    --scope staged_file --input /app/moros_pipeline/output/incoming_new.csv \
    --tier flash --concurrency 400 --estimated-usd <X> --confirm
#   -> moros_pipeline/output/incoming_new_classification_events.csv
```

Resumable (already-classified records are never re-paid for), crash-safe (each event streams to
disk), and locked (a second run on the same events file exits naming the holder).

### 3. Build documents and load them

Citations and licences are fetched for the batch by three keys (`pmid -> doi -> pmcid`), then the
staged CSV and its events become documents through the same `schema.build_document()` every
record in the corpus was built with.

```bash
cd moros_pipeline/scripts
python3 fetch_citations.py --with-licence --input ../output/incoming_new.csv
python3 join_citations.py --citations ../output/epmc_citations.csv \
    --with-licence --corpus-from-moros --output ../output/pid_licences.csv
python3 ../../mongo_landscape_export/scripts/build_staged_documents.py \
    --staged ../output/incoming_new.csv \
    --events ../output/incoming_new_classification_events.csv
#   -> ../../mongo_landscape_export/output/incoming_new_documents.jsonl

python3 load_documents.py --input ../../mongo_landscape_export/output/incoming_new_documents.jsonl --dry-run
python3 load_documents.py --input ... --limit 100 --confirm     # real, reversible trial
python3 load_documents.py --input ... --confirm
python3 ensure_indexes.py && python3 verify_corpus.py --expect-count <previous + loaded>
```

Records with no abstract, and parse errors, are left out and counted rather than guessed at.

**After any load, by hand:** restart `observatory-ws` (its facet cache is boot-loaded with no
TTL, so new journals, MeSH terms and licences are invisible until then), and if the document
shape changed, cut a schema release in `dome-ml-observatory` with its `schema-version` skill.

### 4. Enrichment — tag the positives with controlled vocabularies

A separate, deliberate, costly cycle: roughly 13,000x the per-record price of classification,
run per journal or cohort when wanted. The system message (prompt `e1`) carries the three
vocabularies in `curation_criteria/` verbatim: EDAM domain in three tiers, learning paradigm and
model family (closed), and canonical model-type spellings (open: unlisted methods are tagged
verbatim). It never asks the positive/negative question, so it cannot revise a verdict, and the
database writer's allowlist makes that structural. Records already enriched are excluded from the
export. About 1.5% of records hit the 16,000-token output cap and are recorded as truncated
rather than retried blindly.

```bash
cd moros_pipeline/scripts
python3 export_journal_for_enrichment.py --journal "Bioinformatics (Oxford, England)" \
    --out ../output/enrich_input_<name>.csv           # names are matched VERBATIM; --max-usd guards it
docker compose run --rm pipeline dome-triage llm-classify enrich \
    --tier flash --concurrency 800 \
    --input /app/moros_pipeline/output/enrich_input_<name>.csv \
    --events-out /app/moros_pipeline/output/enrichment_<name>_events.csv   # ALWAYS pass --events-out
python3 load_enrichment.py --events ../output/enrichment_<name>_events.csv --dry-run
python3 load_enrichment.py --events ... --limit 25 --confirm
python3 load_enrichment.py --events ... --confirm
```

Measured cost: **$4.07 per 1,000 records**, 94% of it output tokens, of which 98% is the model's
reasoning trace. That is not reducible: lowering reasoning effort cost more and produced six times
the vocabulary violations; disabling thinking was 88% cheaper and agreed with production on all
six fields for 0% of records. Throughput is `concurrency / ~45 s per call`: about 418 records per
minute at 800.

### 5. Citation refresh — keep `citation_count` current on existing records

Counts are stored with a fetch timestamp, so a refresh only re-fetches entries older than a chosen
age and writes three fields through the `citations` allowlist, which cannot reach anything else.

```bash
cd moros_pipeline/scripts
python3 fetch_citations.py --max-age-days 30                  # only stale keys; lite result type
python3 join_citations.py --corpus-from-moros                 # -> ../output/pid_citations.csv
python3 load_fields.py --mode citations --input ../output/pid_citations.csv            # dry run
python3 load_fields.py --mode citations --input ../output/pid_citations.csv --limit 500 --confirm
python3 load_fields.py --mode citations --input ../output/pid_citations.csv --confirm
```

### 6. Cross-links — scaffold only

`identifiers.dome_registry`, `bioai_repo`, `huggingface`, `kaggle` and `zenodo` exist on every
document and are null. [`cross_links/`](cross_links/README.md) holds the plan and a scaffold for
filling them from Europe PMC data links, software links and repository APIs. Nothing there runs
yet.

## Costs and timing

[`COST_DASHBOARD.md`](COST_DASHBOARD.md) is the one-page answer: what a monthly classification
batch costs, what enriching the remaining positives costs, at DeepSeek V4 Flash off-peak and peak
rates and at GLM-5.3-Flash list rates, from measured token profiles and the live corpus counts.
Regenerate it with `python3 scripts/cost_dashboard.py --live --balance`; the pricing it reads is
[`pricing/pricing.yaml`](pricing/pricing.yaml), dated and sourced.

DeepSeek bills **double during 01:00–04:00 and 06:00–10:00 UTC, Monday to Friday**; every other
hour, and all weekend, is off-peak. `python3 scripts/offpeak_window.py --minutes <N>` says whether
a run of that length started now stays off-peak, and when the next such window opens.

## Repository map

| Path | What it is |
|---|---|
| `moros_pipeline/` | Everything that talks to moros, and the fetch/dedupe/citation/licence machinery. Its [README](moros_pipeline/README.md) is the refresh runbook and the record of what was measured. |
| `mongo_landscape_export/` | The document builders and the reference copy of the document shape ([README](mongo_landscape_export/README.md)). |
| `src/dome_triage/`, `tests/`, `pyproject.toml`, `docker/` | The validated classification/enrichment engine, carried unchanged. Only `llm-classify classify | enrich | calibrate | project-cost` are operational here; the other subcommands are inert. |
| `curation_criteria/` | The maintained assets the prompts are built from: `CRITERIA.md`, the three vocabulary JSONs, the validation fixtures. |
| `prompts/` | The exact rendered system messages with their hashes, and the script that regenerates and verifies them ([README](prompts/README.md)). |
| `schema/` | Alignment with the published schema in `dome-ml-observatory` ([README](schema/README.md)). |
| `pricing/`, `scripts/cost_dashboard.py`, `COST_DASHBOARD.md` | Pricing as of a date, the generator, and the dashboard. |
| `epmc_licensing/` | The original pmid-keyed licence table (13 MB), still read as a fallback by the document builder. |
| `cross_links/` | Scaffold for the reserved external-identifier fields. |
| `docs/` | Specifications for work not yet built. [`preprint.md`](docs/preprint.md) is the Europe PMC preprint-venue capture and backfill (schema v1.3.0). |
| `.claude/skills/`, [`SKILLS.md`](SKILLS.md) | Agent skills, one per process, plus the sequential refresh cycle. |
| [`AGENTS.md`](AGENTS.md) | The rules: what may never be altered, how writes are kept safe, what has gone wrong before. |
| [`ROADMAP.md`](ROADMAP.md) | Short. |

## Maintained assets and what "validated" means

The prompts, the vocabularies, the document schema, the pid rule and the write allowlists are the
validated processing. They are versioned by content hash (criteria sha256, vocab sha256,
`schema_version`) and every record in the corpus carries the hashes it was produced under. Editing
any of them is a versioned change with a re-validation, never a tweak; see `AGENTS.md`. The
engine's own docstrings cite the research documents (step logs, ablations) in which each number
was measured; those documents are the scientific record and are not needed to operate this
pipeline.

## Licence and citation

CC BY 4.0 on everything this repository adds (code, prompts, vocabularies, schema, decisions and
rationales), see [`LICENSE.md`](LICENSE.md). The bibliographic metadata and abstracts come from
Europe PMC and keep their own terms; `source.access.license` on each document records them. Cite
with [`CITATION.cff`](CITATION.cff).
