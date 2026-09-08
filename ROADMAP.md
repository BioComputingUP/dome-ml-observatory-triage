# Roadmap

1. **Preprint data through the pipeline.** Author schema v1.3.0 (`publication_metadata.
   preprint_server`, `source.epmc_source`, `identifiers.epmc_id`), harvest the three on every run,
   and backfill the ~56,863 existing preprint records, per [`preprint.md`](preprint.md). One cheap
   `lite` fetch; clears the standing `check_alignment.py` drift.
2. **Finalise schema versioning.** With v1.3.0 authored, settle the release procedure — who bumps,
   when, what a release carries — so the authored `SCHEMA_VERSION`, the published `schema/CURRENT`
   and the live `schema_version` stop drifting. `schema/check_alignment.py` is the arbiter.
3. **Test and build this repository** end to end — `pytest`, a clean install, `ruff` — with the
   skills in `.claude/skills/` exercised in tandem against the commands they document.
4. **Cross links.** Build the fetch process for the reserved `identifiers.*` fields (Hugging Face,
   DOME Registry, Zenodo, `bioai_repo`, Kaggle) per [`cross_links/README.md`](cross_links/README.md),
   then add an `identifiers` write mode to `moros_write.py` with tests.
5. **Automated Zenodo bulk push.** Monthly GitHub Actions workflow: cursor-loop `GET /api/export`,
   gzip, deposit through the Zenodo API with a sidecar (count, size, sha256, `schema_version`) and
   the schema release. Reuse `DOME_zenodo_archive/download_dome_registry.py`, `ZENODO_TOKEN` from
   Actions secrets. Replaces the unregistered DOI hardcoded on `/download/bulk`.
6. **DCAT integration.** Plan how the corpus and its releases are described as a DCAT dataset/
   distribution, and where that description is served.
7. **schema.org / JSON-LD.** Plan a JSON-LD equivalent of the record schema and database
   metadata, kept in step with `schema/`.
8. **Test 6 and 7** end to end against the live service before either is published.
9. **Keep the two repositories aligned.** Last, once the items above land: `check_alignment.py`
   clean, and no claim, count or link in either repo's `README.md`, `ROADMAP.md` or skills
   contradicting the other's. The sister roadmap carries the matching item.

Short list, unranked, to judge later:

- Settle the refresh cadence (monthly or bimonthly) as stated policy, and make the skills, the
  front end's claimed update frequency and a visible processing-log provenance on the site agree.
- Investigate Europe PMC data links and ELIXIR Core Data Resource / EBI resource relations as
  corpus signals — adjacent to 3, wider than the `identifiers.*` fields.
- Enrich the remaining positives (~360k with an abstract; see `COST_DASHBOARD.md`), by journal
  cohorts, off-peak, weekends.
- Decide whether `citation_count` needs an index (`ensure_indexes.py --measure-citation-sort`).
- Any move to GLM-5.3-Flash (or any model) needs re-validation against the human benchmark and the
  enrichment agreement check before it replaces DeepSeek V4 Flash; the dashboard column is a price
  comparison, not a validated option.
- Search-space expansion terms measured worth adding (BERT, GPT, U-Net, ResNet, YOLO); each starts
  fresh coverage under its own query hash.
- Archive each batch's event logs (the paid record) somewhere durable; they are gitignored here.
- `model_type` is open-vocabulary and captures tool names alongside methods; decide facet policy.
- Automate the post-load `observatory-ws` restart and facet-stats check.
