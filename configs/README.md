# configs/

Declared settings and paths, so no step hardcodes either. Five files are loaded together on
**every** CLI command (`cli.py::_load_config` names them explicitly — nothing scans this folder,
so adding a file like this README changes nothing). They must stay parseable even when a run does
not use them.

| File | What it declares | Used by this repo's operational commands |
|---|---|---|
| [`sources.yaml`](sources.yaml) | two halves: `paths:`, every input and output file the pipeline reads or writes, and `label_sources`/`fulltext_roots`, the external datasets the original labelled set was built from | **the `paths:` half, yes** — it resolves the event logs and calibration log. The source registry is inert here |
| [`pipeline.yaml`](pipeline.yaml) | the spend cap and its log, plus curation and keyword output paths | **the `budget:` block, yes** — `classify` checks cumulative spend against `deepseek_second_curator.total_cap_usd` before every paid run |
| [`sampling.yaml`](sampling.yaml) | stratified sampling bands and the bulk-pool paths | no |
| [`tfidf.yaml`](tfidf.yaml) · [`keybert.yaml`](keybert.yaml) | keyword-extraction settings | no |
| [`curation_features.yaml`](curation_features.yaml) | structured flags the curation app captured per decision | no — and it is **not** one of the five: the app reads it directly |

**Paths never name a machine.** Output paths are relative to the repository root. External data is
`${DOME_TRIAGE_DATA_ROOT}/...`, which defaults to this repository's parent directory and is
expanded by `config.py::resolve_path`, raising if a referenced variable is unset. See `AGENTS.md`.

**The inert files are kept deliberately.** They are loaded on every command, so removing one breaks
every command, and they document the shape the classified corpus was built under.
