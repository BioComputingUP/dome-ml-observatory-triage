# schema/

Where the document shape is authored, where it is published, and the checks that keep the two and
the database in agreement.

| | Location | Role |
|---|---|---|
| **Authored** | this repo: `mongo_landscape_export/scripts/schema.py` (`SCHEMA_VERSION`, `build_document`), its generated reference `mongo_landscape_export/schema/ai_ml_landscape.schema.json`, and the vocabularies `curation_criteria/{domain,modelling_branch,model_type_seed}_vocab.json` | the shape every document is built with |
| **Published** | `dome-ml-observatory/schema/` : `CURRENT`, immutable `releases/vX.Y.Z/` (JSON Schema + example + `vocab/`), `CHANGELOG.md`, `validate.py` | the versioned contract the UI and API build against |
| **Held** | moros, `schema_version` on every document | what is actually live |

`llm_classification.classification` stays the single classification field: the `positives_text`
partial index and the API's `$text` guard key on it.

## Release procedure

Both repositories follow this procedure. `check_alignment.py` enforces what it can, the sibling's
`schema-version` skill cuts the release, and this repository's `schema-sync` skill says which step is
next.

**What counts as a release.** Any change a published release carries: the document shape
(`build_document()`'s output) or any of the three vocabulary files. Either one bumps
`SCHEMA_VERSION` and restamps every document on moros, so authored, published and live always name
the same version. A vocabulary-only change is a patch (v1.5.1), an added field a minor (v1.4.0,
v1.6.0), and a removed, renamed or retyped field a major; the sibling's `schema/README.md` has the
full semver rules.

**Who moves first.** This repository bumps `SCHEMA_VERSION`, then the sibling cuts the release in the
same change set. The published side is never ahead of the authored side, and moros is never ahead
of either.

**The order:**

1. **Author** here: `schema.py` (bump `SCHEMA_VERSION` and add the dated history comment), run
   `write_schema_template.py`, change the vocabulary files if they changed, and write the migration.
   Even a stamp-only release needs `migrate_vX_Y_Z.py` and its `WRITE_MODES` entry. Run the tests and
   `python3 prompts/render_prompts.py --check`.
2. **Publish** in `dome-ml-observatory` with its `schema-version` skill. That means the new immutable
   `schema/releases/vX.Y.Z/` folder (JSON Schema, a real example, vocabularies copied verbatim), a
   `CHANGELOG.md` entry with a migration note, `CURRENT`, and both `FALLBACK_SCHEMA_VERSION`
   constants. Then refresh `observatory_release/` here with the `schema-sync` skill.
3. **Check.** `python3 schema/check_alignment.py` must say `aligned`.
4. **Commit** both repositories.
5. **Migrate** moros: `migrate_vX_Y_Z.py`, a dry run and then `--confirm`. This comes before any field
   load, because the field modes never write `schema_version`, so values loaded first would sit
   under the old version.
6. **Load** any new field values with `load_fields.py --mode ...`: dry run, trial, confirm.
7. **Verify.** Run `ensure_indexes.py` (`--confirm` if the release adds an index), then
   `verify_corpus.py`, then `check_alignment.py --live`, which must say `aligned`.
8. **Deploy** the sibling's apps and restart `observatory-ws`. An additive release can deploy before
   the migration, since the apps read both shapes.
9. **Close out.** Replace the CHANGELOG entry's "Pending at release" with what ran and when, update
   the status line at the end of this file, and commit.

**What a release carries.** Unless a row says otherwise, `check_alignment.py` checks it:

| Where | What | Checked by |
|---|---|---|
| here | `SCHEMA_VERSION`, and the template's `schema_version` | `check_alignment.py` |
| here | `WRITE_MODES["migrate_vX_Y_Z"]` and `migrate_vX_Y_Z.py` | `check_alignment.py` (the mode) |
| here | `observatory_release/` byte-identical to the published release | `check_alignment.py` |
| here | the version `verify_corpus.py` expects | nothing to move: it reads `schema.py` |
| there | the release's field paths, data-link element keys and vocabularies | `check_alignment.py` |
| there | the release example's `schema_version` | `check_alignment.py` |
| there | a `## vX.Y.Z` entry in `schema/CHANGELOG.md` | `check_alignment.py` |
| there | `CURRENT` and both `FALLBACK_SCHEMA_VERSION` constants | `check_alignment.py` |
| there | the code that mirrors the shape: `record.model.ts`, `record.schema.ts`, and `PROJECTED_PATHS` in `observatory-ws/src/metadata/record-view.ts` | their own tests (`PROJECTED_PATHS` is tested against `CURRENT`) |
| moros | every document at the version | `check_alignment.py --live`, `verify_corpus.py` |

Corpus releases are a separate thing from schema releases, with their own description:
[`docs/release_metadata.md`](../docs/release_metadata.md).

## The check

```bash
python3 schema/check_alignment.py                 # authored vs published (../dome-ml-observatory)
python3 schema/check_alignment.py --live          # also reads schema_version off moros (read-only)
python3 schema/check_alignment.py --observatory-dir /path/to/dome-ml-observatory
(cd schema && python3 -m pytest .)                # its own tests, over a fake pair of repositories
```

It exits 1 on any difference, naming each one with the step that fixes it.

`observatory_release/` is a snapshot of the published release this repository was last aligned
with, copied by the `schema-sync` skill so a diff is readable offline. It is a copy, not an
authority.

**Where the three sides stand (2026-09-15).** 1.6.0 is on all three. It is authored here and
published as v1.6.0 (the snapshot is in `observatory_release/`). `migrate_v1_6_0.py` stamped all
846,716 documents on moros at 20:09 UTC, and `record_modified_positive` is built.
`verify_corpus.py` passes and `check_alignment.py --live` reports `aligned`. 1.6.0 adds the
top-level `record_modified` datestamp, which OAI-PMH and the sitemaps page by.
