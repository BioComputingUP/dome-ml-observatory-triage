# AGENTS.md

Instructions for AI agents and people working in this repository. Read it fully before changing
anything. If it disagrees with the code, trust the code and fix this file.

## What this is

The write side of DOME Observatory: the pipeline that fetches, classifies, enriches, formats and
loads AI/ML publication metadata into `dome_observatory.Content` on the MongoDB host **moros**.
The read side, `dome-ml-observatory` (UI + API + the published schema releases), is a sibling
repository and is read-only by design; **it must never gain write credentials, and this
repository must never serve reads.**

`README.md` explains each process. `SKILLS.md` lists the agent skills. `ROADMAP.md` is short on
purpose.

## Ground rules

1. **Validated processing is carried, not edited.** The following are the scientific method of
   this corpus and were validated before being brought here. Changing any of them is a versioned
   change (new prompt/vocab/schema version, changelog, re-validation), never an in-place tweak:
   - `curation_criteria/CRITERIA.md` and `src/dome_triage/llm_classify/prompts.py` (classification
     prompt `v1`, criteria sha256 `bd9d66dd…`, kappa 0.81 against human curation);
   - `curation_criteria/*.json` and `src/dome_triage/llm_classify/enrichment.py` (enrichment prompt
     `e1`, vocab sha256 `41db952f…`);
   - `mongo_landscape_export/scripts/schema.py` (`SCHEMA_VERSION`, the document shape) and
     `pid.py` (the UUID5 `_id` rule);
   - `moros_pipeline/scripts/moros_write.py` (`WRITE_MODES`, the per-mode field allowlists, and
     `STAMPS_RECORD_MODIFIED`, the modes that move `record_modified`);
   - the parsers, the resumption keys and the event-log columns.
   `python3 prompts/render_prompts.py --check` fails if a prompt no longer renders to its recorded
   hash; run it before and after touching anything above.

2. **Every database write is dry-run by default, allowlisted, and reversible.** All writes go
   through `moros_write.py::SafeWriter` or `load_documents.py`. `--confirm` is required to write;
   `--limit N` makes a real, reversible N-document trial first; a rollback snapshot of exactly the
   paths about to change is written to `moros_pipeline/output/rollback/<run_id>.jsonl` before the
   first batch. There is no delete, drop or whole-document replace path, and there must not be one.
   **Never drop `Content`**: it destroys `positives_text`, and while that index is gone the
   service silently degrades to a regex scan with no error anywhere.

3. **Money is gated.** `classify` requires `--estimated-usd` and `--confirm`, checks the cumulative
   spend in `data/processed/deepseek_second_curator_spend_log.csv` against the cap in
   `configs/pipeline.yaml`, and logs every run. `enrich` has no gate by design, so the gate is
   `export_journal_for_enrichment.py --max-usd` and the person running it. Before any paid run:
   pull the live price and balance (`cost-estimate` skill), state the projection, and confirm.
   Prefer off-peak (see `scripts/offpeak_window.py`): peak is 01:00–04:00 and 06:00–10:00 UTC,
   Monday to Friday, at double the rate.

4. **One events file per batch, always `--events-out`.** A staged batch's `record_id` is the Mongo
   `_id` (UUID5). Appending to a log from a different identifier space breaks resumption for both.
   Both `classify` and `enrich` hold an exclusive `flock` on their events file; a second run exits
   naming the holder. Pass `--ignore-lock` only for a container the daemon has genuinely lost.

5. **A detached `docker compose run` exits non-zero while its container keeps working.** The
   service sets `tty: true`. Never retry on that exit code: judge progress by the output file and
   start a new container only when `docker ps` confirms none is running. Ten concurrent
   containers once enriched the same records and duplicated $9.63 of paid work.

6. **Two authorities, not merged.** The coverage ledger knows what was *fetched*; moros knows what
   was *loaded*. The ledger may be ahead. If moros is ahead, `fetch_search_space.py` stops. Do not
   add an option that reconciles this silently. The ledger in this repository is the live one from
   2026-09-07; there is no other copy to consult.

7. **The database is the system of record.** Event logs and built JSONL under
   `moros_pipeline/output/` and `mongo_landscape_export/output/` are gitignored. An events file is
   the paid record of a run and is merged into moros with `load_documents.py` /
   `load_enrichment.py` before it is ever discarded.

8. **Nothing in the container touches the database; nothing on the host imports `dome_triage`.**
   Keep that split. The container is for the DeepSeek-calling CLI; host Python is for moros.

9. **Docker only for the engine.** No venv, no host `pip install` of the package: the model weights
   and NLTK corpora are baked into the image and a host run would silently diverge. The gate is
   `docker compose run --rm pipeline pytest --ignore=tests/test_curate_page.py` (that one file
   drives the inert Streamlit app and cannot resolve its script path from this repository root).
   Every operational test must pass; if one fails, stop and report rather than patching validated
   code. The three host suites are `(cd moros_pipeline/scripts && python3 -m pytest .)`,
   `(cd mongo_landscape_export/scripts && python3 -m pytest .)` and `(cd schema && python3 -m pytest .)`. Rebuilding the
   image leaves the previous one dangling at ~7 GB; run `docker image prune -f` after a few builds
   and check `df -h /`. Never `docker system prune -a` without asking.

## Runtime layout the verbatim scripts expect

Folder names are load-bearing. `build_incoming_documents.py` imports `pid.py` from
`../../mongo_landscape_export/scripts`; `build_staged_documents.py` defaults its inputs to
`<repo>/epmc_licensing/output/epmc_pmid_licensing.csv`, `<repo>/moros_pipeline/output/
epmc_licence_backfill.csv` and `<repo>/moros_pipeline/output/epmc_citations.csv`; the engine
reads `curation_criteria/` from the repository root. Do not rename or move these folders.

**No tracked file may contain a machine-specific absolute path.** A `/home/<user>/...` path
makes the repository unrunnable for anyone else and fails silently rather than loudly; eighteen
config lines and sixty-two ledger entries carried one until 2026-09-07. Config refers to external
data as `${DOME_TRIAGE_DATA_ROOT}/...`, resolved by `config.py::resolve_path`, which defaults the
root to this repository's parent and RAISES on an unset variable rather than resolving somewhere
unintended. Container paths (`/app/...`) are portable and fine. Run
`python3 scripts/check_no_absolute_paths.py` before committing; `tests/test_config_paths.py` pins
the behaviour.

The engine's docstrings cite documents from the research project in which it was validated
(`STEPS_Progress.md`, `Database_STEPS_Progress.md`, `FINALISATION_ROADMAP.md`,
`thinking_ablation/`, `enrichment_trial_dataset/`). They are not in this repository and are not
needed to run it; the measured numbers they carry are restated in `README.md`, `moros_pipeline/
README.md` and `COST_DASHBOARD.md`. `configs/sources.yaml` likewise still lists absolute paths of
research inputs that only the inert `ingest` subcommands read.

`enrich` defaults its input to a research trial set that is not here: **always pass `--input`**
(the export from `export_journal_for_enrichment.py`).

Three more defaults point at staging files that are not here, so the skills always pass the
explicit alternative: `fetch_citations.py --input` (use `scripts/export_corpus_keys.py`'s output
for a refresh, or the staged CSV for a batch), `join_citations.py --corpus-from-moros`, and
`build_curated_documents.py`'s canonical/bulk inputs (the frozen curated set; supply them
explicitly if that path is ever re-run).

`moros_pipeline/output/epmc_citations.csv` and `epmc_licence_backfill.csv` are the resumable
fetch state (gitignored, ~97 MB): `--max-age-days` decides what is stale by reading them. Keep
them; if they are lost, the next refresh simply re-fetches everything (free, ~17 minutes).

## After any load — the manual checklist

1. Restart `observatory-ws` in the sibling repository's deployment. Its `FacetsService` is
   boot-loaded with no TTL; `StatsService`, `CountService` and `JournalsService` are 24h TTL.
2. In `dome-ml-observatory`, `python3 schema/generate_facet_stats.py --from-api <url>` and check
   `/api/stats` reconciles with `verify_corpus.py`.
3. If `SCHEMA_VERSION` or a vocabulary changed: `python3 schema/check_alignment.py` here, then
   cut the release in `dome-ml-observatory` with its `schema-version` skill. Never hand-edit a
   published `schema/releases/vX.Y.Z/` there.

## Schema alignment with `dome-ml-observatory`

The document shape is **authored here** (`schema.py`, `write_schema_template.py`,
`curation_criteria/*.json`) and **published there** (`schema/releases/`, `schema/CURRENT`,
`CHANGELOG.md`). Both must agree with what moros actually holds. `schema/check_alignment.py`
compares the three (`--live` reads `schema_version` off moros) and exits non-zero on drift. Run it
before any load and whenever a skill in either repository touches the schema; both repositories'
`AGENTS.md` name this obligation, so neither side changes the shape without the other knowing.
The `positives_text` index is partial on `llm_classification.classification == "positive"` and
`observatory-ws` gates `$text` queries on exactly that field: keep it the single classification
field.

**The release procedure** -- what counts as a release (the shape or any vocabulary), who moves first
(this repository), the order of publish, migrate, load, verify and deploy, and every place a release
writes its version -- is in [`schema/README.md`](schema/README.md). `check_alignment.py` checks each
of those places and names any that disagrees. Monthly corpus releases are described separately, in
DCAT and schema.org, by `build_release_metadata.py` ([docs/release_metadata.md](docs/release_metadata.md)).

## Things that have gone wrong before — do not reintroduce

- **A pmid-only licence fetch left 74,472 documents at `license: null`** (68,103 had no pmid).
  Fetch by `pmid -> doi -> pmcid`; write `""` (looked up, none disclosed) not `null` (never looked
  up) for a key EPMC cannot answer, or every future backfill re-fetches it forever.
- **`mongoimport` cannot complete a load against this server** (stalls at ~17%, reports 0 imported
  after landing 1,000). `load_documents.py` uses pymongo `ReplaceOne(upsert=True)` and treats the
  collection count as the only honest authority.
- **A whole-document replace blanks whatever the new document does not carry.** `load_documents.py`
  refuses to replace an existing document whose content differs; that guard is what kept 23
  enriched curated papers from being wiped. Keep it.
- **Truncated enrichment responses retried at the same cap re-truncate deterministically.** Raise
  `ENRICHMENT_MAX_TOKENS` first, then `--retry-truncated`.
- **A too-low `--concurrency` is invisible until hours have passed.** Enrichment at 40 crawled at
  53 records/min; 800 gives ~418. The HTTP pool is sized from the flag.
- **`verify_corpus.py` once matched one batch prefix and ignored 13,476 new documents.** Invariants
  are about populations, not batch names.
- **A stale stats key crashed a builder after printing its success line and before writing.** Check
  the artifact's mtime and count, not the log line.
- **`docker compose` created a root-owned empty `.git/` before this repository was a git
  repository.** `docker-compose.yml` mounts `./.git:/app/.git:ro` so `provenance.py` can read the
  commit; Docker creates a missing bind-mount source as a root-owned directory, and `git init`
  then fails with `.git/hooks/: Permission denied`. Harmless once the repository exists (2026-09-07:
  `rmdir .git`, then `git init`). If you ever clone into a fresh directory and run Compose before
  the clone finishes, that is the symptom.
- **Europe PMC's text-mined identifiers were stored verbatim and 1,513 documents got links that
  404** (2026-09-14): `10.5281/zenodo.18675888.`, `10.6084/m9.figshare.24123303”.`, DOIs cut from
  comma lists, `×` for `x`, `zenodo.XXXXX` placeholders, free text where an RRID was matched. Never
  store an annotation's `exact` string. `moros_pipeline/scripts/link_identifiers.py` defines a clean
  identifier; `build_data_links.py` is the only place identifiers are chosen (repairing what it can
  and confirming every DOI at doi.org); `load_fields.py --mode data_links` and `load_documents.py`
  refuse a file with a malformed link before connecting; `verify_corpus.py` fails if one is in
  moros. Keep all four in place, and extend `link_identifiers.py` rather than cleaning anywhere else.
- **An EBI Search cross-reference answer without `size` carries one reference per entry, whatever
  its `referenceCount`** (2026-09-14: pdbekb for PMID 33024307 returned 1 of 5), and EBI Search
  returns curation databases beside real deposits. `fetch_ebisearch_xrefs.py detail` checks every
  entry against its count; `moros_pipeline/scripts/ebisearch_resources.py` is the only accept list
  for EBI Search domains, and [`docs/data_links_sources.md`](docs/data_links_sources.md) must
  agree with it. EBI Search links go to positives only and pass the same four gates.

## Provenance of this repository

The scripts, package, prompts and vocabularies were validated in a research project and carried
here byte-for-byte on 2026-09-07 (commit `e553279` of that project's history, recorded here for
reproducibility only; it is not a dependency of anything in this repository). This repository is
the operational home from that date. On 2026-09-15 the two modelling vocabularies gained
`ontology_mappings` ([docs/vocabulary_ontology_mappings.md](docs/vocabulary_ontology_mappings.md)); the
rendered enrichment prompt is unchanged.
