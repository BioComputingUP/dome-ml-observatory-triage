# Roadmap

1. **DCAT integration.** Plan how the corpus and its releases are described as a DCAT dataset/
   distribution, and where that description is served.
2. **schema.org / JSON-LD.** Plan a JSON-LD equivalent of the record schema and database
   metadata, kept in step with `schema/`.
3. **Test 1 and 2** end to end against the live service before either is published.

Short list, unranked, to judge later:

- **Author schema v1.3.0** in `mongo_landscape_export/scripts/schema.py` — `publication_metadata.
  preprint_server`, `source.epmc_source`, `identifiers.epmc_id` — per [`docs/preprint.md`](docs/preprint.md),
  then migrate the 846,716 documents in place and backfill the ~56,863 preprint records. The
  published schema went **ahead** of the authored one on 2026-09-07 (`CURRENT` v1.3.0, database and
  `schema.py` both 1.2.0); `schema/check_alignment.py` names the three fields. Nothing populates
  them yet, so no data is wrong — the shape is simply not authored here yet.
- Fill the reserved `identifiers.*` cross-link fields (see `cross_links/`), then add an
  `identifiers` write mode to `moros_write.py` with tests.
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
