---
name: schema-sync
description: >
  Check that the document schema authored in this repository, the schema published by the sibling
  dome-ml-observatory repository, and the live schema_version on moros all agree, and say which
  side has to move when they do not. Trigger on "is the schema aligned", "check the schema",
  "the vocab changed", "sync the schema", before any load, and whenever a skill in either
  repository touches schema/. Read-only; it never edits a published release.
---

# schema-sync

The shape is **authored here** (`mongo_landscape_export/scripts/schema.py` → `SCHEMA_VERSION`,
`build_document()`; `curation_criteria/*.json`), **published there**
(`dome-ml-observatory/schema/releases/vX.Y.Z/`, `CURRENT`, `CHANGELOG.md`) and **held** on moros
(`schema_version` on every document). `schema/README.md` here and `AGENTS.md` in both
repositories state the same rules.

## Procedure

```bash
python3 schema/check_alignment.py --live      # add --observatory-dir <path> if the sibling is not at ../dome-ml-observatory
```

Read the report. `aligned` → done, say so. Otherwise, for each named difference:

- **Version drift (authored ahead of published).** The release has to be cut in
  `dome-ml-observatory` with its `schema-version` skill: new immutable `releases/v<authored>/`
  folder, vocab files copied verbatim from `curation_criteria/` (renamed to `domain.json`,
  `modelling-branch.json`, `model-type-seed.json`), a real example record, a `CHANGELOG.md` entry
  with a migration note, `CURRENT` moved, the UI vocab re-synced, their `validate.py` passing.
  Never do that by hand-editing a published folder, and never do it from this repository.
- **Fields authored but not published** — same as above; list them for the changelog. Also flag
  the three places there that mirror the shape (`record.model.ts`, `record.schema.ts`,
  `records.query.ts` + `search-params.ts`).
- **Vocab drift** — additive `parent_ids`-style metadata still needs publishing (a minor bump);
  a changed term set is also a new enrichment `vocab_sha256`, so every existing enrichment run
  resumes as a fresh batch. Say both.
- **Live documents behind the authored version** — an in-place migration is due here, with the
  same shape as `migrate_v1_2_0.py` (constant values, allowlisted, reversible). Never a reload.
- **Published ahead of authored** — should not happen; the observatory does not author. Stop and
  report.

After a release lands there, refresh the offline snapshot here and commit it:

```bash
rm -rf schema/observatory_release && mkdir -p schema/observatory_release
cp -r ../dome-ml-observatory/schema/releases/v<X.Y.Z> schema/observatory_release/
cp ../dome-ml-observatory/schema/CURRENT schema/observatory_release/CURRENT
python3 schema/check_alignment.py
```

## Changing the shape (when the user asks for it)

1. Edit `schema.py`, bump `SCHEMA_VERSION`, run `python3 write_schema_template.py` and the tests in
   `mongo_landscape_export/scripts/`.
2. Add any new writable path to `moros_write.py::WRITE_MODES` on purpose, with a test.
3. Plan the migration of existing documents (pattern: `migrate_v1_2_0.py`).
4. Publish there (above). 5. `check_alignment.py --live` must say `aligned` before the next load.

Keep `llm_classification.classification` the single classification field: `positives_text` is a
partial index on it and the API's `$text` guard depends on it.
