# schema/

Where the document shape is authored, where it is published, and the check that keeps the two and
the database in agreement.

| | Location | Role |
|---|---|---|
| **Authored** | this repo: `mongo_landscape_export/scripts/schema.py` (`SCHEMA_VERSION`, `build_document`), its generated reference `mongo_landscape_export/schema/ai_ml_landscape.schema.json`, and the vocabularies `curation_criteria/{domain,modelling_branch,model_type_seed}_vocab.json` | the shape every document is built with |
| **Published** | `dome-ml-observatory/schema/` : `CURRENT`, immutable `releases/vX.Y.Z/` (JSON Schema + example + `vocab/`), `CHANGELOG.md`, `validate.py` | the versioned contract the UI and API build against |
| **Held** | moros, `schema_version` on every document | what is actually live |

Rules, shared with the sibling repository's `AGENTS.md`:

- A shape change starts **here** (`schema.py`, bump `SCHEMA_VERSION`, `write_schema_template.py`,
  tests) and is **published there** with its `schema-version` skill, which copies the vocab files
  verbatim and writes the changelog entry. Never hand-edit a published release folder.
- Both repositories run `check_alignment.py` before a load or a release. Drift is a stop, not a
  warning.
- `llm_classification.classification` stays the single classification field: the `positives_text`
  partial index and the API's `$text` guard key on it.

## The check

```bash
python3 schema/check_alignment.py                 # authored vs published (../dome-ml-observatory)
python3 schema/check_alignment.py --live          # also reads schema_version off moros (read-only)
python3 schema/check_alignment.py --observatory-dir /path/to/dome-ml-observatory
```

It compares: `SCHEMA_VERSION` here vs `CURRENT` there; the field paths of the reference template
here vs the published JSON Schema there; each vocabulary JSON here vs the published `vocab/` copy
(JSON equality, ignoring key order); and with `--live`, the `schema_version` values moros holds.
Exit code 1 on any difference, with each one named.

`observatory_release/` is a snapshot of the published release this repository was last aligned
with, copied by the `schema-sync` skill so a diff is readable offline. It is a copy, not an
authority.

**The snapshot is v1.1.0 and deliberately behind.** As of 2026-09-07 the sibling repository has
uncommitted `releases/v1.2.0/` and `releases/v1.3.0/` folders and a `CURRENT` of v1.3.0. A
snapshot is only ever taken from a committed release, so this one waits. `check_alignment.py`
reads the sibling working tree directly and already reports the real state: published v1.3.0,
authored and live 1.2.0, three fields (`identifiers.epmc_id`,
`publication_metadata.preprint_server`, `source.epmc_source`) published but not yet authored here.
Authoring them is the first item in `ROADMAP.md`; the specification is `docs/preprint.md`.
