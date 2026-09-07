# curation_criteria/

The maintained assets the classification and enrichment prompts are built from. Nothing here is
generated: each file is a decision someone made, and each is read at runtime by
`src/dome_triage/llm_classify/`.

**The rulebook — what counts as relevant**

- [`CRITERIA.md`](CRITERIA.md) — the definition of positive, negative and undeterminable. Its full
  text goes verbatim into the classification prompt, and every classification event records the
  sha256 of the exact version it was judged under, so an edit starts a fresh, non-conflated batch.
- [`example_decision_cases.csv`](example_decision_cases.csv) — 14 real papers with the answer the
  rulebook should give and the rule each one tests, chosen as the hard cases either side of the
  boundary: classical machine learning (positive), a PRISMA systematic review, informal chatbot
  use, and plain regression (all negative). `llm-classify validate-criteria` scores the prompt
  against these, which is the cheap check to run before any paid run and after any edit above.
- [`QUEUE_CONSTRUCTION.md`](QUEUE_CONSTRUCTION.md) — how the human curation queue was assembled.

**The controlled vocabularies — how a relevant paper is tagged**

Read together into the enrichment prompt, whose `vocab_sha256` is the hash of the rendered text, so
editing any of them correctly starts a fresh enrichment batch.

- [`domain_vocab.json`](domain_vocab.json) — the subject domain, three tiers from the EDAM topic
  branch. Closed.
- [`modelling_branch_vocab.json`](modelling_branch_vocab.json) — learning paradigm and model
  family, from the NLM MeSH AI/ML branch. Closed. Decided by hand; its reasoning is in the file's
  own `decision_note`.
- [`model_type_seed_vocab.json`](model_type_seed_vocab.json) — 76 canonical method spellings with
  aliases. **Open**: a method not listed is tagged verbatim, so this normalises spelling rather
  than restricting the answer.

Editing any file here is a versioned change with a re-validation, never a tweak — see `AGENTS.md`.
`prompts/` holds the rendered prompts these produce, with a check that they still hash as expected.
