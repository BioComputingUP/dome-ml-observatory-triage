# e3 vocabulary gaps, and what to do at load time

> **Status 2026-11-23:** the load-time clean-up this note asks for now exists. It is
> `moros_pipeline/scripts/enrichment_clean.py` (rule set c1), applied by `load_enrichment.py` to every
> event, and the generated gap list is [`enrichment_vocab_gaps.md`](enrichment_vocab_gaps.md). Read that
> one before a load. This note stays as the reasoning behind `curation_criteria/candidates/`.

For: whoever merges the e3 enrichment events into moros (the ingestion laptop), and whoever picks
up the vocabulary next. Written 2026-11-20, after the e3 production run finished. **Read this before
running `moros-write` section B on any e3 events file.**

Decision (Gavin, 2026-11-20): **e3 stays the production prompt; no e4 now.** Model-type gaps are
kept as candidates in [`curation_criteria/candidates/`](../curation_criteria/candidates/README.md).
The stray domain tags below are documented here only: **no clean-up code has been built yet.**

## The run

| | core_bioinformatics | batch2_high_impact |
|---|---:|---:|
| records | 13,065 | 87,159 |
| parsed (`ok`) | 13,042 | 87,031 |
| `parse_error` (all cut at the 16,000-token cap) | 23 | 128 |

Prompt e3 (`vocab_sha256 3b9d119d…`), GLM-5.3-Flash on the EMBL-EBI gateway, 2026-11-16 to
2026-11-20. Events: `moros_pipeline/output/enrichment_e3/<cohort>/*_events.csv` here, and in the
private runs repository (`events/<cohort>/`, commit `c50dde2`, `SHA256SUMS` verified byte-for-byte
against the local files at the end of the run). 2,493 parsed records (2.5%) carry at least one
`vocab_violations` entry.

## What a flag means, stage by stage

| Stage | What happens to a flagged value |
|---|---|
| Parser (`src/dome_triage/llm_classify/enrichment.py` `parse_enrichment`, `enrichment_e2.py` `parse_e2`) | A domain term in the wrong tier is **moved** to its real tier when there is room (`…:repaired_to_…`). Everything else is **kept as the model returned it** and listed in `vocab_violations` ("tolerate and log, never silently dropped") |
| Loader (`moros_pipeline/scripts/load_enrichment.py` `event_to_update`) | Writes the tags as they are into `content_filters.*` and the flag list into `llm_enrichment.vocab_violations`. Never reads `domain_vocab.json`. After `--confirm` it only prints a count |
| `verify_corpus.py` | Does not check domain values |
| observatory-ws (sibling repo) | Facets and stats use raw distinct values (`facets.service.ts`, `stats.service.ts`): **a stray term becomes a visible facet option.** JSON-LD (`record-jsonld.mapper.ts`) publishes it as a `DefinedTerm` with no EDAM IRI but still labelled as part of the domain vocabulary |

## The flags, by kind

| Kind | Count | Verdict |
|---|---:|---|
| `model_type:excluded_kind:optimiser_check` (genetic algorithm 1,179, PSO 362, Bayesian optimisation 172, grey wolf 99) | 2,025 | **Useful, keep.** A review marker (rule E2-V8): in a 10-paper sample, 6 were the paper's own method (legitimate), 4 a helper (feature selection, tuning) |
| other `excluded_kind` (task 123, simulation 100, component 68, pipeline 5, explainability 5, statistics 3, family_name 2) | 306 | Mixed; fine to keep flagged |
| `domain_tier*:unknown` | 236 | **Bad: reach the website.** See below |
| `domain_tier*:cap_exceeded` | 35 | Harmless but over the vocabulary's limit |
| `…:repaired_to_…` | 63 | Already fixed by the parser |
| `model_family:unknown` (`reinforcement`, a paradigm) | 6 | Minor |

Model types outside the seed are never flagged (`model_type` is open): 34.7% of tags. The
candidates for normalising them are in `curation_criteria/candidates/`.

## Stray domain tags (the part that needs action)

Measured over the 100,073 parsed events; 267 records (0.27%) would change under the rules below.

- **Real terms in the wrong tier, home tier already full — 226.** The model wanted a third tier-2
  topic and put it in tier 3 (or a tier-1 term in tier 2). Top: tier 3 holding `Sequence analysis`
  32, `Biomarkers` 28, `Data mining` 25, `Structure analysis` 23, `Sequence sites, features and
  motifs` 16, `Proteomics experiment` 10, `Proteomics` 9, `Genomics` 9; tier 2 holding `Laboratory
  techniques` 11, `Omics` 9. Example: *On the performance of pre-microRNA detection algorithms* —
  tier 2 `[Bioinformatics, Genomics]`, tier 3 `[Functional, regulatory and non-coding RNA, Sequence analysis]`.
- **Typos — 3:** tier 1 `Omic`, tier 2 `Hematology` (vocab: `Haematology`), tier 2 `Cheminformics`.
- **Not in any tier — 7:** `Rehabilitation medicine` ×4, `Ophthalmology`, `Geology`, `Nucleic acids
  sites, features and motifs`.
- **Over the cap — 35 lists:** 32 tier-3 lists of 4 (cap 3), 3 tier-2 lists of 3 (cap 2).

### The deterministic clean-up rules (for when it is built)

Apply per record, in this order, using `curation_criteria/domain_vocab.json` only:

1. A value that is a label of its own tier: keep, in the model's order.
2. Same label ignoring case: replace with the canonical label.
3. Reviewed typo map, and nothing fuzzier: `Omic → Omics`, `Hematology → Haematology`,
   `Cheminformics → Cheminformatics`. Extend it only by review, never by edit distance.
4. A label of another tier: move it to its home tier if that tier is under its cap and does not
   already hold it; otherwise drop it. Process tier 1, then tier 2, then tier 3, after steps 1–3, so
   a dropped invalid value can free a slot (seen once: tier 2 `Omics` dropped, tier 3 `Genomics` moved up).
5. Anything else: drop.
6. Truncate each tier to `max_tags` (1 / 2 / 3), keeping the model's order.

Simulated on these events (read-only): 225 dropped (home tier full), 7 dropped (not in vocab),
33 truncated, 3 typo-mapped, 1 moved; **0 records with any non-vocab value or over-cap list afterwards.**
The cost is real: those 225 drops are topics the model judged relevant that the cap does not allow.

How it must be built, so nothing rubbish gets in and nothing is lost:

- **Never edit an events file** (it is the paid record). Clean at load time.
- **Keep the raw value:** append each action to `llm_enrichment.vocab_violations` as
  `clean:<tier>:<action>:<value>` (`moved`, `dropped`, `typo`, `case`, `truncated`).
- **One place cleans:** a pure module (copy the shape of `moros_pipeline/scripts/link_identifiers.py`:
  `clean()`, `problems()`, an empty list is the only acceptable answer).
- **The loader refuses before connecting** if any non-vocab domain value would remain (copy
  `load_fields.py` / `load_documents.py`'s "refusing to load …, nothing written" gate).
- **`verify_corpus.py` fails** while any document holds a domain value outside the vocabulary
  (`$nin` the allowed labels, per tier; tier 1 is a scalar, tiers 2–3 are arrays).
- **Documents already in moros** (earlier e1/e2 loads may hold strays too) are repaired through the
  existing `enrichment` write mode: dry run first, `--limit`, rollback snapshot.
- Tests for every rule, including order (rule 4) and the empty-after-clean case.

### Decide on the ingestion laptop, before loading

- **Load as-is now:** the e3 tags reach the site today; about 0.27% of records carry a stray domain
  value, visible as odd facet options and in stats, until a repair pass.
- **Build the clean-up first** (rules above), then load: nothing stray ever reaches moros. A few
  hours of work plus tests; no LLM calls, no cost.

## Load checklist for these two cohorts (moros laptop)

1. Pull the runs repository; `sha256sum -c SHA256SUMS`.
2. `python3 schema/check_alignment.py --live`; if moros is below 1.7.0, `migrate_v1_7_0.py` first
   (`load_enrichment.py` refuses e3 events until then — they carry `model_use`).
3. Per chunk, `moros-write` section B: `load_enrichment.py --events …` (dry run), then
   `--limit 50 --confirm`, then `--confirm`. `parse_error` rows are skipped automatically.
4. `verify_corpus.py`, then restart observatory-ws and refresh facet stats (AGENTS.md
   "After any load").

The 151 records cut at the token cap stay unenriched. Retrying them is not a flag: the cap is
`ENRICHMENT_MAX_TOKENS = 16000` in `enrichment.py` (frozen, recorded in `PROMPT_HASHES.json`), so
raising it is a deliberate versioned change, then `--retry-truncated` (AGENTS.md). Retrying at the
same cap re-truncates.

## Seen during the run, not fixed

- `scripts/checkpoint_events.py` dies on a network error (DNS during a Wi-Fi drop, 2026-11-19
  22:23); its loop only survives a refused push. Backups stopped for 2.5 h until restarted.
- The laptop watchdog has no "backups stopped" alert; the GitHub dead-man check is the only one,
  and GitHub ran its 20-minute cron only a few times a day.
- Provenance records `git 87fdef8-dirty` although `git status` was clean on the host.
- `enrich_status.py` shows "truncated 0" while every parse error was a length cut.
- The sibling's observatory-ws container was in a restart loop on 2026-11-18.
- EBI VPN sessions lasted 12–14 h; the "extend session" prompt does not always appear.
