"""Asserts the corpus's invariants against the live collection, and fails loudly when one breaks.

This is F5 from `FINALISATION_ROADMAP.md` -- the part that stops the original defect recurring.
That defect was not a bug in any one script: a deliberate exclusion simply left no artifact anyone
downstream would trip over, so 6,197 records went missing from the published corpus and it took a
spot-check on a famous paper to notice. The fix has three parts, and this file is the third:

1. the exclusion now writes `data/processed/landscape_excluded_curated.csv` (see
   `build_curated_documents.py`),
2. the conversion report reconciles the counts and the loader refuses to run if it did not,
3. **and this asserts, against the live collection, that the populations add up.**

The headline invariant is the one that would have caught it on day one:

    countDocuments() == documents with provenance "llm"
                      + documents with provenance "human_curated"
                      + documents with provenance "registry_confirmed"

with every provenance value non-null. Any future refresh that silently drops a whole population
fails here instead of shipping.

The AlphaFold probe is the acceptance test the finalisation roadmap set: the four queries in its
section 7 returned 0 hits each, on DOI, PMID, PMCID and a title regex. They must now return 1,
badged `registry_confirmed`.

    python3 verify_corpus.py
    python3 verify_corpus.py --expect-count 833235
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from moros_client import Moros
from link_identifiers import MONGO_MALFORMED_REGEX
import resolve_duplicates as rd
from moros_write import (
    RECORD_MODIFIED_PATH,
    RECORD_MODIFIED_PATTERN,
    REPORT_DIR,
    new_run_id,
    utc_now_iso,
)

SCHEMA_PY = Path(__file__).resolve().parents[2] / "mongo_landscape_export" / "scripts" / "schema.py"


def authored_schema_version(path: Path = SCHEMA_PY) -> str:
    """`SCHEMA_VERSION` read out of schema.py rather than repeated here: a second literal is one more
    place a release has to remember to move (schema/README.md, "Release procedure")."""
    m = re.search(r'^SCHEMA_VERSION\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"SCHEMA_VERSION not found in {path}")
    return m.group(1)


EXPECTED_SCHEMA_VERSION = authored_schema_version()
VALID_PROVENANCE = ("llm", "human_curated", "registry_confirmed")
VALID_CLASSIFICATIONS = ("positive", "negative", "undeterminable")
REQUIRED_INDEXES = ("_id_", "class_year_id", "positives_text", "record_modified_positive")
# Versions older than v1.5.0, which introduced the EBI Search link keys.
PRE_V1_5_0 = ("1.1.0", "1.2.0", "1.3.0", "1.4.0")

# Jumper et al., Nature 596 (2021). The paper whose absence exposed the whole gap.
ALPHAFOLD = {
    "doi": "10.1038/s41586-021-03819-2",
    "pmid": "34265844",
    "pmcid": "PMC8371605",
    "title": "Highly accurate protein structure prediction with AlphaFold",
    # The finalisation roadmap's fourth probe was a substring regex it recorded as returning 0.
    # Measured on the live collection, it actually returns 3 -- three *different* papers legitimately
    # contain the phrase, one of them Tunyasuvunakool et al.'s "...for the human proteome", which
    # that same document lists as a present-and-positive neighbour. So the substring was never an
    # identity test. The exact title is; the substring is kept as an informational count that
    # should go 3 -> 4 when the merge lands.
    "title_regex": "Highly accurate protein structure prediction",
}


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append((name, passed, detail))

    def note(self, name: str, detail: str) -> None:
        self.rows.append((name, None, detail))

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.rows if r[1] is False]

    def render(self) -> None:
        width = max(len(r[0]) for r in self.rows)
        for name, passed, detail in self.rows:
            mark = "  --  " if passed is None else (" PASS " if passed else " FAIL ")
            print(f"[{mark}] {name:<{width}}  {detail}")


def verify(moros: Moros, expect_count: int | None) -> tuple[Checks, dict]:
    checks = Checks()
    facts: dict = {}

    total = moros.count()
    facts["total"] = total
    print(f"target {moros.describe()}\n")

    # -- schema version ------------------------------------------------------
    versions = moros.histogram("schema_version")
    facts["schema_version"] = {str(k): v for k, v in versions.items()}
    checks.add(
        "schema_version uniform",
        list(versions) == [EXPECTED_SCHEMA_VERSION],
        f"{facts['schema_version']}",
    )

    # -- the headline invariant ---------------------------------------------
    provenance = moros.histogram("source.decision_provenance")
    facts["decision_provenance"] = {str(k): v for k, v in provenance.items()}
    accounted = sum(v for k, v in provenance.items() if k in VALID_PROVENANCE)
    checks.add(
        "populations account for every document",
        accounted == total,
        f"{accounted:,} of {total:,} accounted for; {facts['decision_provenance']}",
    )
    checks.add(
        "no document lacks a provenance",
        None not in provenance,
        f"{provenance.get(None, 0):,} with a null/absent decision_provenance",
    )
    checks.add(
        "no unexpected provenance value",
        all(k in VALID_PROVENANCE for k in provenance),
        f"values: {sorted(str(k) for k in provenance)}",
    )
    if expect_count is not None:
        checks.add("document count as expected", total == expect_count,
                   f"{total:,} (expected {expect_count:,})")
    else:
        checks.note("document count", f"{total:,}")

    # -- classification -----------------------------------------------------
    classification = moros.histogram("llm_classification.classification")
    facts["classification"] = {str(k): v for k, v in classification.items()}
    checks.add(
        "classification values all valid",
        all(k in VALID_CLASSIFICATIONS for k in classification),
        f"{facts['classification']}",
    )

    # -- the check that would have caught the original defect ---------------
    batches = moros.histogram("llm_classification.batch_id")
    facts["batch_ids"] = {str(k): v for k, v in batches.items()}
    # Anything not a curated merge is a machine batch. Matching the original Step 23a prefix
    # literally would under-report every batch added since -- the incremental loop stamps
    # `classify_flash_staged_file_*`, and the first one landed 13,476 documents that this check
    # would silently have ignored. The invariant is about populations, not one batch's name.
    curated_batches = {k: v for k, v in batches.items() if k and k.startswith("curated_merge_")}
    llm_batches = {k: v for k, v in batches.items()
                   if k and not k.startswith("curated_merge_")}
    checks.add(
        "corpus contains BOTH populations",
        bool(curated_batches) and bool(llm_batches),
        f"{len(llm_batches)} LLM batch(es) covering {sum(llm_batches.values()):,} docs, "
        f"{len(curated_batches)} curated batch(es) covering {sum(curated_batches.values()):,}",
    )

    # -- one paper, one document --------------------------------------------
    #
    # The `_id` is UUID5 of the first of pmcid > doi > pmid, so the same paper fetched before and
    # after Europe PMC gave it a PMCID mints two ids. The 2026-09-03 load compared only `_id` and
    # inserted 10,572 second copies; `build_incoming_documents.py` has checked the identifiers since
    # 2026-09-15, and this is what makes a recurrence loud rather than silent. Groups that are not
    # one paper twice -- two Europe PMC records sharing a DOI, or a curated record -- are reported,
    # not failed. The scan costs a couple of minutes on the full corpus.
    duplicate_groups = rd.duplicate_groups(moros)
    decided = [rd.classify_group(docs) for docs in duplicate_groups.values()]
    removable = [d for d in decided if d["kind"] in ("plain", "merge_then_remove")]
    left: dict[str, int] = {}
    for decision in decided:
        if decision["kind"] in rd.SKIP_KINDS:
            left[decision["kind"]] = left.get(decision["kind"], 0) + 1
    facts["duplicates"] = {"groups": len(duplicate_groups),
                           "removable_groups": len(removable),
                           "documents_removable": sum(len(d["losers"]) for d in removable),
                           "left_alone": left}
    checks.add(
        "no paper is in the corpus twice",
        not removable,
        f"{len(removable):,} group(s) hold one paper under two _ids "
        f"({sum(len(d['losers']) for d in removable):,} document(s)) -- run resolve_duplicates.py; "
        f"{sum(left.values()):,} group(s) left alone by design: {left}",
    )

    # -- citations ----------------------------------------------------------
    with_count = moros.count({"publication_metadata.citation_count": {"$ne": None}})
    negative = moros.count({"publication_metadata.citation_count": {"$lt": 0}})
    undated = moros.count({
        "publication_metadata.citation_count": {"$ne": None},
        "publication_metadata.citation_count_updated": None,
    })
    facts["citations"] = {"populated": with_count, "negative": negative, "undated": undated,
                          "coverage_pct": round(with_count / total * 100, 2)}
    checks.note("citation coverage",
                f"{with_count:,} of {total:,} ({facts['citations']['coverage_pct']}%) -- the "
                f"remainder is genuine EPMC coverage, null meaning 'not available'")
    checks.add("no negative citation counts", negative == 0, f"{negative:,} negative")
    checks.add("every populated count is dated", undated == 0,
               f"{undated:,} counts with no citation_count_updated")

    # -- licences -----------------------------------------------------------
    #
    # Tracked as an invariant because the gap went unnoticed once already. `null` here means
    # "never looked up", distinct from `""` ("looked up, EPMC disclosed none"), and 74,472
    # documents sat in that state on 2026-09-03 -- 68,103 of them because the original fetcher
    # was pmid-keyed and they have no pmid, so they were structurally unfetchable rather than
    # genuinely unlicensed. A number that grows again means a load skipped the licence step.
    never = moros.count({"source.access.license": None})
    none_disclosed = moros.count({"source.access.license": ""})
    real = total - never - none_disclosed
    facts["licences"] = {"real": real, "none_disclosed": none_disclosed, "never_looked_up": never}
    checks.note("licence coverage",
                f"{real:,} with a licence, {none_disclosed:,} looked up and none disclosed, "
                f"{never:,} never looked up")
    checks.add("licence gap is closed", never == 0,
               f"{never:,} documents have never had a licence lookup -- run "
               f"fetch_citations.py --with-licence --targets moros:missing-licence")

    # -- Europe PMC identity and preprint server (v1.3.0) -----------------
    #
    # Populated by `load_fields.py --mode preprints` off the `fetch_epmc_metadata.py` pass. Every
    # document should carry epmc_id / epmc_source once that has run; a preprint without a server
    # name afterwards is a record to investigate, not a null to accept (docs/preprint.md).
    preprint_filter = {"content_filters.pub_types": {"$in": ["Preprint", "preprint"]}}
    with_epmc_id = moros.count({"identifiers.epmc_id": {"$ne": None}})
    epmc_sources = moros.histogram("source.epmc_source")
    preprints = moros.count(preprint_filter)
    preprints_no_server = moros.count({**preprint_filter,
                                       "publication_metadata.preprint_server": None})
    facts["epmc_identity"] = {"with_epmc_id": with_epmc_id, "epmc_source": epmc_sources,
                              "preprints": preprints,
                              "preprints_without_server": preprints_no_server}
    checks.note("epmc_id coverage",
                f"{with_epmc_id:,} of {total:,} ({with_epmc_id / total * 100:.2f}%); "
                f"epmc_source {epmc_sources}")
    checks.note("preprints without a preprint_server",
                f"{preprints_no_server:,} of {preprints:,} preprints -- expected 0 once the "
                f"preprints backfill has run")

    # -- data links (v1.4.0) -----------------------------------------------
    #
    # `has_data: null` means never looked up (the summary pass has not reached the record);
    # `fetched_at: null` means no link fetch yet. Both are coverage notes, not invariants: a
    # partial backfill is a valid corpus state.
    dl_looked_up = moros.count({"data_links.has_data": {"$ne": None}})
    dl_has_data = moros.count({"data_links.has_data": True})
    dl_fetched = moros.count({"data_links.fetched_at": {"$ne": None}})
    dl_with_resources = moros.count({"data_links.resources.0": {"$exists": True}})
    facts["data_links"] = {"looked_up": dl_looked_up, "has_data": dl_has_data,
                           "fetched": dl_fetched, "with_resources": dl_with_resources}
    checks.note("data links summary coverage",
                f"{dl_looked_up:,} looked up, {dl_has_data:,} with Europe PMC data")
    checks.note("data links fetched",
                f"{dl_fetched:,} fetched, {dl_with_resources:,} carrying at least one resource")
    # Europe PMC's text-mined strings carry the punctuation and words around them; stored verbatim
    # they made links that 404 (1,513 documents on 2026-09-14). build_data_links.py repairs them and
    # both loaders refuse them, so this must stay zero. ~2.5 s over the corpus.
    malformed = moros.count({"$or": [
        {"data_links.links.id": {"$regex": MONGO_MALFORMED_REGEX}},
        {"data_links.links.url": {"$regex": MONGO_MALFORMED_REGEX}},
    ]})
    facts["data_links"]["malformed_documents"] = malformed
    checks.add("no malformed data link identifiers", malformed == 0,
               f"{malformed:,} documents carry a link id or url with stray punctuation, whitespace or "
               f"non-ASCII -- rebuild with build_data_links.py and reload --mode data_links")

    # -- EBI Search links and the DOME Registry identifier (v1.5.0) ---------------
    #
    # Coverage notes, and one invariant: a document carrying v1.5.0 link keys under another
    # schema_version was loaded before migrate_v1_5_0.py ran (load_fields never writes the version).
    with_ebisearch = moros.count({"data_links.sources": "ebisearch"})
    dome_found = moros.count({"identifiers.dome_registry": {"$nin": [None, ""]}})
    dome_none = moros.count({"identifiers.dome_registry": ""})
    facts["data_links"]["ebisearch"] = with_ebisearch
    facts["dome_registry"] = {"found": dome_found, "looked_up_none": dome_none}
    checks.note("EBI Search route", f"{with_ebisearch:,} documents had EBI Search consulted")
    checks.note("identifiers.dome_registry",
                f"{dome_found:,} with a DOME Registry entry, {dome_none:,} looked up with none")
    early = moros.count({"data_links.links.matched_by": {"$exists": True},
                         "schema_version": {"$in": list(PRE_V1_5_0)}})
    checks.add("no v1.5.0 link keys under an older schema_version", early == 0,
               f"{early:,} documents -- run migrate_v1_5_0.py")

    # -- record_modified (v1.6.0) ----------------------------------------------
    #
    # Every document carries one, in the one format OAI-PMH's from/until range queries compare as
    # strings. A missing or malformed value hides the record from incremental harvesters.
    unstamped = moros.count({RECORD_MODIFIED_PATH: {"$not": {"$regex": RECORD_MODIFIED_PATTERN}}})
    facts["record_modified"] = {"missing_or_malformed": unstamped}
    checks.add("every document has a well-formed record_modified", unstamped == 0,
               f"{unstamped:,} documents without a YYYY-MM-DDThh:mm:ssZ record_modified -- run "
               f"migrate_v1_6_0.py")

    # -- indexes ------------------------------------------------------------
    indexes = sorted(moros.indexes())
    facts["indexes"] = indexes
    missing = [i for i in REQUIRED_INDEXES if i not in indexes]
    checks.add(
        "all required indexes present",
        not missing,
        f"{indexes}" + (f"  MISSING {missing}" if missing else ""),
    )

    # -- enrichment ---------------------------------------------------------
    enriched = moros.count({"llm_enrichment.provider": {"$ne": None}})
    facts["enriched"] = enriched
    checks.note("enriched documents", f"{enriched:,}")

    # -- the acceptance probe ----------------------------------------------
    probes = {
        "doi": moros.count({"identifiers.doi": ALPHAFOLD["doi"]}),
        "pmid": moros.count({"identifiers.pmid": ALPHAFOLD["pmid"]}),
        "pmcid": moros.count({"identifiers.pmcid": ALPHAFOLD["pmcid"]}),
        "exact_title": moros.count({"publication_metadata.title": ALPHAFOLD["title"]}),
    }
    substring_hits = moros.count({"publication_metadata.title":
                                  {"$regex": ALPHAFOLD["title_regex"], "$options": "i"}})
    facts["alphafold_probes"] = probes
    facts["alphafold_title_substring_hits"] = substring_hits
    checks.add(
        "AlphaFold 2 present on all four identity probes",
        all(v == 1 for v in probes.values()),
        f"{probes} (every one was 0 before this work)",
    )
    checks.note(
        "AlphaFold title substring",
        f"{substring_hits} paper(s) contain the phrase -- 3 other papers legitimately do, so this "
        f"is a neighbourhood count, not an identity check",
    )
    doc = moros.find_one({"identifiers.pmid": ALPHAFOLD["pmid"]})
    if doc:
        prov = doc["source"].get("decision_provenance")
        cls = doc["llm_classification"].get("classification")
        cites = doc["publication_metadata"].get("citation_count")
        facts["alphafold"] = {"provenance": prov, "classification": cls, "citation_count": cites,
                              "schema_version": doc.get("schema_version")}
        checks.add("AlphaFold 2 badged registry_confirmed", prov == "registry_confirmed",
                   f"provenance={prov}, classification={cls}, citations={cites}")
        checks.add("AlphaFold 2 is a searchable positive", cls == "positive",
                   f"enters positives_text only as classification='positive' -> {cls}")
    else:
        checks.add("AlphaFold 2 retrievable", False, "not found by pmid")

    return checks, facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-count", type=int, default=None)
    args = parser.parse_args()

    run_id = new_run_id("verify_corpus")
    with Moros.from_env() as moros:
        checks, facts = verify(moros, args.expect_count)
        print()
        checks.render()

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{run_id}.report.json"
        path.write_text(json.dumps({
            "run_id": run_id, "target": moros.describe(), "finished_at": utc_now_iso(),
            "facts": facts,
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks.rows],
            "failures": [n for n, p, _ in checks.rows if p is False],
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nreport -> {path}")

        print("\nPost-load checklist -- these are manual and none of them happen on their own:")
        print("  1. Restart observatory-ws. FacetsService is boot-loaded with NO TTL, so new")
        print("     journals, MeSH terms and licences never appear until it restarts.")
        print("     StatsService / CountService / JournalsService are 24h TTL.")
        print("  2. Re-run  python3 schema/generate_facet_stats.py --from-api <url>  in")
        print("     ../dome-ml-observatory, then check /api/stats reconciles.")
        print("  3. If SCHEMA_VERSION or a vocabulary changed: python3 schema/check_alignment.py")
        print("     --live here, then cut the release there with the schema-version skill.")

        if checks.failures:
            raise SystemExit(f"\n{len(checks.failures)} invariant(s) FAILED -- see above.")
        print("\nAll invariants passed.")


if __name__ == "__main__":
    main()
