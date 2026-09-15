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

The shape is authored in this repository, published in the sibling and held on moros:

- **Authored:** `mongo_landscape_export/scripts/schema.py` (`SCHEMA_VERSION`, `build_document()`)
  and `curation_criteria/*.json`.
- **Published:** `dome-ml-observatory/schema/releases/vX.Y.Z/`, `CURRENT` and `CHANGELOG.md`.
- **Held:** the `schema_version` on every document in moros.

Both repositories follow the release procedure in `schema/README.md` ("Release procedure"). This
skill checks where things stand against it and names the next step.

## Procedure

```bash
python3 schema/check_alignment.py --live      # add --observatory-dir <path> if the sibling is not at ../dome-ml-observatory
```

If the report says `aligned`, say so and stop. Otherwise each named difference maps to one step of
the procedure:

- **Version drift, authored ahead, or fields, data-link keys or vocabularies authored but not
  published.** This is step 2: cut the release in `dome-ml-observatory` with its `schema-version`
  skill. List the differences for the changelog. Never cut it from here, and never hand-edit a
  published folder. If a vocabulary's term set changed, that also means a new enrichment
  `vocab_sha256`, so every existing enrichment run resumes as a fresh batch. Say so.
- **Published ahead of authored.** The procedure forbids this. Stop and report.
- **No `migrate_vX_Y_Z` mode.** Step 1 is incomplete: every release restamps moros, even one that
  changes only the vocabularies.
- **Template, release example, CHANGELOG entry or `FALLBACK_SCHEMA_VERSION` disagrees.** The release
  is incomplete on that side. Name the file.
- **Snapshot stale.** Refresh it (below).
- **Live documents not at the authored version.** This is step 5: run the release's migration, as a
  dry run and then `--confirm`. Never a reload.

After a release lands in the sibling, refresh the offline snapshot here and commit it:

```bash
rm -rf schema/observatory_release && mkdir -p schema/observatory_release
cp -r ../dome-ml-observatory/schema/releases/v<X.Y.Z> schema/observatory_release/
cp ../dome-ml-observatory/schema/CURRENT schema/observatory_release/CURRENT
python3 schema/check_alignment.py
```

## Changing the shape or a vocabulary (when the user asks for it)

Follow `schema/README.md`'s procedure. The parts done in this repository:

1. In `schema.py`, bump `SCHEMA_VERSION` with a dated history comment. Run
   `python3 write_schema_template.py` and the tests in `mongo_landscape_export/scripts/`.
2. Add any new writable path to `moros_write.py::WRITE_MODES` on purpose, with a test. If the path
   is a value the Observatory's metadata publishes (Dublin Core, JSON-LD), the mode that writes it
   belongs in `STAMPS_RECORD_MODIFIED`: a harvester can only see a change if `record_modified`
   moves with it.
3. Write `migrate_vX_Y_Z.py` and its `WRITE_MODES` entry. Use `migrate_v1_5_1.py` as the pattern for
   a version stamp, and `migrate_v1_6_0.py` for a stamp plus a constant new field.
4. Publish in the sibling, refresh the snapshot, confirm `check_alignment.py` says `aligned`, and
   commit both repositories.
5. Migrate, load, verify (`--live` must say `aligned`), deploy, and close out.

Keep `llm_classification.classification` the single classification field: `positives_text` is a
partial index on it, and the API's `$text` guard depends on it.
