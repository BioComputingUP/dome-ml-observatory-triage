# mongo_landscape_export

The document builders: everything that turns a classified record into a `dome_observatory.Content`
document of the current schema shape, plus the checked-in reference copy of that shape.

The folder name is kept because `moros_pipeline/scripts/build_incoming_documents.py` imports
`pid.py` from here by relative path, and `build_staged_documents.py` resolves its default input
paths relative to the repository root. Moving or renaming anything here changes validated code.

| File | What it does |
|---|---|
| `scripts/schema.py` | `build_document(row)` / `build_curated_document(row)`: one CSV row in, one grouped, typed document out. Owns `SCHEMA_VERSION` and the three `decision_provenance` values. Pure functions, no I/O. |
| `scripts/pid.py` | Mints the deterministic UUID5 `_id` (`pmcid > doi > pmid`). The same paper always mints the same `_id`, which is what makes every load idempotent. **Never reimplement this.** |
| `scripts/citations_index.py` | Loads a `fetch_citations.py` output and looks a record up by `pmid -> doi -> pmcid`. |
| `scripts/build_staged_documents.py` | Staged CSV + classification events + citations/licences -> documents JSONL. The recurring path. |
| `scripts/build_curated_documents.py` | The human-curated / registry-confirmed path. Its inputs (the frozen curated dataset) are not in this repository; it is retained because its tests pin the shared document shape. |
| `scripts/join_license.py` | Defines the `load_licensing` contract the builders read the pmid licence table through. |
| `scripts/write_schema_template.py` | Regenerates `schema/ai_ml_landscape.schema.json` from `schema.py`. Run after any shape change. |
| `schema/ai_ml_landscape.schema.json` | One empty document, every field present: the reference of the shape moros holds. The published, versioned releases live in the sister repository `dome-ml-observatory/schema/releases/`. |
| `scripts/test_*.py` | Hermetic tests. `test_schema.py::test_both_builders_produce_the_same_shape` is what keeps the two builders from drifting. |

Run the tests from inside `scripts/`:

```bash
cd mongo_landscape_export/scripts && python3 -m pytest .
```

`output/` is gitignored: it receives each batch's `<name>_documents.jsonl` and `.report.json`,
both regenerable from the staged CSV and its events file.
