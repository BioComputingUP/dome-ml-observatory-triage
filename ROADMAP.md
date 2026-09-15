# Roadmap

1. **Test and build this repository** end to end — `pytest`, a clean install, `ruff` — with the
   skills in `.claude/skills/` exercised in tandem against the commands they document.
2. **Automated Zenodo bulk push.** Monthly GitHub Actions workflow: cursor-loop `GET /api/export`,
   gzip, deposit through the Zenodo API with a sidecar (count, size, sha256, `schema_version`) and
   the schema release. Reuse `DOME_zenodo_archive/download_dome_registry.py`, `ZENODO_TOKEN` from
   Actions secrets. Then add the deposit to that month's release metadata as a Zenodo distribution
   with its DOI (`build_release_metadata.py`, [docs/release_metadata.md](docs/release_metadata.md)),
   and replace the unregistered DOI hardcoded on `/download/bulk`.

3. **Keep the two repositories aligned.** Last, once the items above land: `check_alignment.py`
   clean, and no claim, count or link in either repo's `README.md`, `ROADMAP.md` or skills
   contradicting the other's. The sister roadmap carries the matching item.

Short list, unranked, to judge later:

1. Settle the refresh cadence (monthly or bimonthly) as stated policy, and make the skills, the
   front end's claimed update frequency and a visible processing-log provenance on the site agree.
2. ELIXIR Core Data Resource / EBI resource relations as corpus signals, now that `data_links`
   records which resources each paper uses.
3. Enrich the remaining positives (~360k with an abstract; see `COST_DASHBOARD.md`), by journal
   cohorts, off-peak, weekends.
4. Decide whether `citation_count` needs an index (`ensure_indexes.py --measure-citation-sort`).
5. Any move to GLM-5.3-Flash (or any model) needs re-validation against the human benchmark and the
   enrichment agreement check before it replaces DeepSeek V4 Flash; the dashboard column is a price
   comparison, not a validated option.
6. Search-space expansion terms measured worth adding (BERT, GPT, U-Net, ResNet, YOLO); each starts
   fresh coverage under its own query hash.
7. Archive each batch's event logs (the paid record) somewhere durable; they are gitignored here.
8. `model_type` is open-vocabulary and captures tool names alongside methods; decide facet policy.
9. Automate the post-load `observatory-ws` restart and facet-stats check.
