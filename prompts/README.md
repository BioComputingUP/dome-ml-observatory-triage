# prompts/

The exact text sent to the model, as it is, with the hash every record in the corpus carries.

Nothing here is edited by hand. The prompts are **assembled in code** from the maintained assets
in `curation_criteria/`; this folder is the rendered, verifiable copy so a person can read what
the model sees without running Docker.

| Prompt | Version | Assembled by | Static system message (rendered here) | Hash recorded on every event |
|---|---|---|---|---|
| Classification | `v1` | `src/dome_triage/llm_classify/prompts.py::build_prompt` | `classification_system_message.v1.txt` | `criteria_sha256` = sha256 of `curation_criteria/CRITERIA.md` |
| Classification, forced choice | `v1` | `prompts.py::build_forced_choice_prompt` | `classification_forced_choice_system_message.v1.txt` | same |
| Enrichment | `e1` | `src/dome_triage/llm_classify/enrichment.py::build_static_system_text` (`flat` rendering) | `enrichment_system_message.e1.txt` | `vocab_sha256` = sha256 of the rendered system message |

`PROMPT_HASHES.json` records both hashes as rendered from the current assets.

**The user message** is the same for both prompts and is built per record from exactly four
fields, read explicitly by key, so nothing else in a row can ever reach the model:

```
Title: {title}
Journal: {journal}
Year: {year}

Abstract:
{abstract}
```

Settings that are part of the validated configuration: `temperature 0`, JSON response mode,
no tools (`enable_search` is never set on these paths), model `deepseek-v4-flash`,
`max_tokens` 6,000 for classification and 16,000 for enrichment (`ENRICHMENT_MAX_TOKENS`),
provider-default reasoning effort and the `flat` domain rendering for enrichment.

## Regenerate and verify

```bash
python3 prompts/render_prompts.py           # re-render from the current assets, rewrite the files
python3 prompts/render_prompts.py --check   # exit 1 if the committed files or hashes are stale
```

`--check` runs before and after any change to `curation_criteria/` or the two prompt modules.
If it fails, either the assets changed (then this is a new prompt version: bump `PROMPT_VERSION`
or `ENRICHMENT_PROMPT_VERSION`, re-validate, and expect resumption to start a fresh batch) or
someone edited a rendered file by hand (then re-render). A key the renderer never reads
(`parent_ids`, `ontology_mappings`) changes only `vocab_file_sha256`: re-render. The rendered
message and `vocab_sha256` stay the same, so it is not a new prompt version.

Historical hashes in production: classification criteria `bd9d66dd892e6c0a…` (every record
classified since 2026-08-21); enrichment vocab `41db952f15118176…` (every enriched record).
