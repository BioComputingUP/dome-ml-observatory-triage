# Skills

Agent skills live in `.claude/skills/<name>/SKILL.md`. Each covers one process; `refresh-cycle`
runs them in sequence with a check-in before every paid or writing step. Every skill uses only the
validated scripts and commands in this repository, exactly as `README.md` documents them.

| Skill | Use it when | Costs money | Writes to moros |
|---|---|---|---|
| `triage-fetch` | "fetch new papers", "what is new since the last run", "run the triage" | no | no |
| `classify` | "classify the staged batch", "run the classification" | yes | no |
| `moros-write` | "load the batch", "merge the enrichment", "roll back", "check the indexes", any write | no | **yes** |
| `enrich` | "enrich journal X", "enrich the new positives" | yes (high) | via `moros-write` |
| `citations-refresh` | "refresh citation counts", every two months | no | via `moros-write` |
| `cost-estimate` | before any paid step; "what would it cost", "update the dashboard", "best time to run" | no | no |
| `schema-sync` | before a load; "is the schema aligned", "the vocab changed" | no | no |
| `data-links` | "fetch the data links", "EBI Search links", "backfill the preprint servers", "refresh data links" | no | via `moros-write` |
| `cross-links` | working on the reserved `identifiers.*` fields (`dome_registry` is filled by `data-links`; the rest is a scaffold) | no | via `moros-write` |
| `refresh-cycle` | "do the refresh", "run the whole pipeline" | asks first | asks first |
| `processing-log` | the close of a refresh; "log the round", "update the processing history page" | no | no (commits the page in `dome-ml-observatory`) |

The sequential order for a refresh: `triage-fetch` → (ask) → `classify` → `moros-write` (load,
index, verify) → (ask) → `enrich` → `moros-write` (merge) → `citations-refresh` (optional) →
full-text refresh (`fetch_fulltext.py`, `moros-write` C) → `data-links` refresh (when due) →
post-load checklist → Zenodo archive (`scripts/zenodo_archive.py`; ask before publishing) →
`processing-log`. A run is usually started from the parameter block in `BULK_UPDATE.md`, which
says what each choice does.
`cost-estimate` runs inside every paid step; `schema-sync` runs before every load.

Rules every skill follows: dry-run before confirm; `--limit` trial before a full write; pull the
live price and balance before spending, and log the billed balance delta after; prefer off-peak; one events file per batch; never a second
container while one is running; restart `observatory-ws` after a load.
