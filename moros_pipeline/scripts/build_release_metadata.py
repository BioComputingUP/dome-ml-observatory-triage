"""Builds the corpus's DCAT 3 / schema.org description for one monthly release, and writes it into
the sister repository, which publishes it.

    python3 build_release_metadata.py                      # dry run: prints the document
    python3 build_release_metadata.py --write              # -> <observatory>/metadata/releases/<YYYY-MM>/
    python3 build_release_metadata.py --report ../output/verify_corpus_<run>.report.json --release 2026-09 --write

**In:** the latest `verify_corpus.py` report -- the counts, and the proof every invariant passed --
the authored `SCHEMA_VERSION`, `prompts/PROMPT_HASHES.json` (the criteria and vocabulary hashes the
classification and enrichment ran under), the search-space query hash, `CITATION.cff` and this
repository's commit. Nothing contacts moros: the report is the record of what moros held.

**Out:** `metadata/releases/<YYYY-MM>/dataset.jsonld` and `metadata/CURRENT` in dome-ml-observatory,
which serves the file at `/api/catalog` and embeds it on its home and bulk-download pages. One
`@graph`: the catalogue, the dataset series (the corpus), this month's dataset in that series, its
API distribution, the API as a data service, and the publisher, creator and contact point. Each node
is typed in both DCAT and schema.org, so a DCAT harvester and a schema.org crawler (Google Dataset
Search) read the same file.

A published month is immutable, like a schema release: an existing month is refused unless
`--overwrite` is given, which is only for a month not yet committed there.

No Zenodo distribution and no DOI yet. The archive job (ROADMAP.md) adds both when it deposits the
month; until then the corpus is identified by its landing page. See docs/release_metadata.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from coverage_ledger import SearchSpace
from moros_write import REPORT_DIR
from verify_corpus import authored_schema_version

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
PROMPT_HASHES = REPO / "prompts" / "PROMPT_HASHES.json"
CITATION = REPO / "CITATION.cff"

ORIGIN = "https://observatory.dome-ml.org"
CATALOG_ID = f"{ORIGIN}/#catalog"
# observatory-ws/src/metadata/metadata-urls.ts::corpusSeriesId names the same node as every record's
# isPartOf. Change the two together or not at all.
SERIES_ID = f"{ORIGIN}/download/bulk#corpus"
API_ID = f"{ORIGIN}/api"
EXPORT_DISTRIBUTION_ID = f"{ORIGIN}/api/export#ndjson"
CONTACT_ID = f"{ORIGIN}/about/support#contact"

PUBLISHER_ID = "https://biocomputingup.it"
UNIVERSITY_ROR = "https://ror.org/00240q980"
CONTACT_EMAIL = "contact@dome-ml.org"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
OBSERVATORY_REPO = "https://github.com/BioComputingUP/dome-ml-observatory"
TRIAGE_REPO = "https://github.com/BioComputingUP/dome-ml-observatory-triage"
DOME_REGISTRY = "https://registry.dome-ml.org"
BIOSCHEMAS_DATASET = "https://bioschemas.org/profiles/Dataset/1.0-RELEASE"
BIOSCHEMAS_DATACATALOG = "https://bioschemas.org/profiles/DataCatalog/0.3-RELEASE-2019_07_01"

CONTEXT = {
    "@vocab": "https://schema.org/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "prov": "http://www.w3.org/ns/prov#",
    "vcard": "http://www.w3.org/2006/vcard/ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

RIGHTS = (
    "CC BY 4.0 covers what DOME Observatory adds: the AI/ML screening verdicts, the controlled-"
    "vocabulary enrichment and the data-link annotations. Titles, abstracts and other bibliographic "
    "metadata come from Europe PMC and remain under Europe PMC's terms of use and each article's "
    "own licence."
)

RELEASE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def release_id(release: str) -> str:
    return f"{ORIGIN}/download/bulk#release-{release}"


def schema_release_url(version: str) -> str:
    return f"{OBSERVATORY_REPO}/tree/main/schema/releases/v{version}"


def latest_report(report_dir: Path = REPORT_DIR) -> Path:
    reports = sorted(report_dir.glob("verify_corpus_*.report.json"))
    if not reports:
        raise SystemExit(f"no verify_corpus_*.report.json in {report_dir} -- run verify_corpus.py first")
    return reports[-1]


def load_report(path: Path, schema_version: str) -> dict[str, Any]:
    """The report, refused unless every invariant passed and every document is at the authored
    schema version: a release must not describe a corpus mid-migration or one that failed a check."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("failures"):
        raise SystemExit(f"{path.name} records failed invariants {report['failures']} -- fix them first")
    facts = report["facts"]
    if facts.get("schema_version") != {schema_version: facts["total"]}:
        raise SystemExit(
            f"{path.name} shows schema_version {facts.get('schema_version')}, not all "
            f"{facts['total']:,} documents at the authored {schema_version} -- migrate, re-verify, rebuild"
        )
    return report


def creators(citation: Path = CITATION) -> list[dict[str, Any]]:
    """The people CITATION.cff credits, as schema.org Persons identified by ORCID where given."""
    data = yaml.safe_load(citation.read_text(encoding="utf-8"))
    out = []
    for author in data.get("authors") or []:
        name = " ".join(p for p in (author.get("given-names"), author.get("family-names")) if p)
        person: dict[str, Any] = {"@type": ["foaf:Person", "Person"], "name": name, "foaf:name": name}
        if author.get("orcid"):
            person["@id"] = author["orcid"]
            person["identifier"] = author["orcid"]
        out.append(person)
    return out


def git_commit(repo: Path = REPO) -> tuple[str, bool]:
    """(HEAD sha, whether the working tree has uncommitted changes)."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def previous_release(observatory_dir: Path, release: str) -> str | None:
    root = observatory_dir / "metadata" / "releases"
    earlier = sorted(p.name for p in root.glob("*") if p.is_dir() and RELEASE_RE.match(p.name)
                     and p.name < release) if root.exists() else []
    return earlier[-1] if earlier else None


def build_document(
    report: dict[str, Any],
    *,
    release: str,
    schema_version: str,
    hashes: dict[str, Any],
    query_sha256: str,
    commit: str,
    people: list[dict[str, Any]],
    previous: str | None = None,
) -> dict[str, Any]:
    """The JSON-LD document, pure: every input is an argument."""
    if not RELEASE_RE.match(release):
        raise ValueError(f"release must be YYYY-MM, got {release!r}")
    facts = report["facts"]
    classes = facts.get("classification", {})
    total = facts["total"]
    positive = classes.get("positive", 0)
    issued = report["finished_at"][:10]
    schema_url = schema_release_url(schema_version)
    creator_refs = [{"@id": p["@id"]} if "@id" in p else p for p in people]

    description = (
        f"Release {release} of the DOME Observatory corpus: {total:,} publications from Europe PMC "
        f"screened against the DOME Observatory curation criteria, {positive:,} of them classified as "
        f"AI/ML methods papers ({classes.get('negative', 0):,} not, "
        f"{classes.get('undeterminable', 0):,} undeterminable). {facts.get('enriched', 0):,} carry "
        f"controlled-vocabulary enrichment (EDAM domains, learning paradigm, model family, model "
        f"type) and {facts.get('data_links', {}).get('with_resources', 0):,} link to research outputs "
        f"such as deposited data, software and DOME Registry entries. Record schema v{schema_version}."
    )

    catalog = {
        "@id": CATALOG_ID,
        "@type": ["dcat:Catalog", "DataCatalog"],
        "dct:conformsTo": {"@id": BIOSCHEMAS_DATACATALOG},
        "name": "DOME Observatory",
        "dct:title": "DOME Observatory",
        "description": "A catalogue of AI/ML methods papers in the life sciences, screened and annotated "
                       "from Europe PMC to support the DOME recommendations for machine-learning reporting.",
        "url": ORIGIN,
        "dcat:landingPage": {"@id": ORIGIN},
        "publisher": {"@id": PUBLISHER_ID},
        "dct:publisher": {"@id": PUBLISHER_ID},
        "license": CC_BY_4,
        "dct:license": {"@id": CC_BY_4},
        "dataset": {"@id": SERIES_ID},
        "dcat:dataset": {"@id": SERIES_ID},
        "dcat:service": {"@id": API_ID},
        "dct:relation": {"@id": DOME_REGISTRY},
    }

    series = {
        "@id": SERIES_ID,
        "@type": ["dcat:DatasetSeries", "Dataset"],
        "dct:conformsTo": {"@id": BIOSCHEMAS_DATASET},
        "name": "DOME Observatory corpus",
        "dct:title": "DOME Observatory corpus",
        "description": "AI/ML methods papers in the life sciences, screened from Europe PMC and annotated "
                       "with controlled vocabularies and linked research outputs; released monthly.",
        "url": f"{ORIGIN}/download/bulk",
        "dcat:landingPage": {"@id": f"{ORIGIN}/download/bulk"},
        "keywords": ["machine learning", "artificial intelligence", "life sciences", "literature",
                     "DOME recommendations", "FAIR"],
        "dcat:keyword": ["machine learning", "artificial intelligence", "life sciences"],
        "creator": creator_refs,
        "dct:creator": creator_refs,
        "publisher": {"@id": PUBLISHER_ID},
        "dct:publisher": {"@id": PUBLISHER_ID},
        "license": CC_BY_4,
        "dct:license": {"@id": CC_BY_4},
        "dct:rights": RIGHTS,
        "isAccessibleForFree": True,
        "includedInDataCatalog": {"@id": CATALOG_ID},
        "hasPart": {"@id": release_id(release)},
        "dcat:last": {"@id": release_id(release)},
        "dcat:contactPoint": {"@id": CONTACT_ID},
    }

    dataset = {
        "@id": release_id(release),
        "@type": ["dcat:Dataset", "Dataset"],
        "dct:conformsTo": [{"@id": BIOSCHEMAS_DATASET}, {"@id": schema_url}],
        "name": f"DOME Observatory corpus, {release} release",
        "dct:title": f"DOME Observatory corpus, {release} release",
        "description": description,
        "dct:description": description,
        "url": f"{ORIGIN}/download/bulk",
        "version": release,
        "dcat:version": release,
        "datePublished": issued,
        "dct:issued": {"@value": issued, "@type": "xsd:date"},
        "isPartOf": {"@id": SERIES_ID},
        "dcat:inSeries": {"@id": SERIES_ID},
        "schemaVersion": schema_url,
        "size": {"@type": "QuantitativeValue", "value": total, "unitText": "records"},
        "variableMeasured": [
            "AI/ML methods-paper classification",
            "EDAM topic (domain tiers 1-3)",
            "learning paradigm",
            "model family",
            "model type",
            "linked research outputs",
        ],
        "creator": creator_refs,
        "dct:creator": creator_refs,
        "publisher": {"@id": PUBLISHER_ID},
        "dct:publisher": {"@id": PUBLISHER_ID},
        "license": CC_BY_4,
        "dct:license": {"@id": CC_BY_4},
        "dct:rights": RIGHTS,
        "isAccessibleForFree": True,
        "includedInDataCatalog": {"@id": CATALOG_ID},
        "distribution": {"@id": EXPORT_DISTRIBUTION_ID},
        "dcat:distribution": {"@id": EXPORT_DISTRIBUTION_ID},
        "dcat:contactPoint": {"@id": CONTACT_ID},
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "name": "DOME Observatory triage pipeline",
            "prov:endedAtTime": {"@value": report["finished_at"], "@type": "xsd:dateTime"},
            "prov:wasAssociatedWith": {
                "@id": f"{TRIAGE_REPO}/tree/{commit}",
                "@type": "SoftwareSourceCode",
                "name": "dome-ml-observatory-triage",
                "codeRepository": TRIAGE_REPO,
                "version": commit,
            },
            "prov:used": [
                {
                    "@type": "CreativeWork",
                    "name": "DOME Observatory curation criteria (classification prompt)",
                    "url": f"{TRIAGE_REPO}/blob/{commit}/curation_criteria/CRITERIA.md",
                    "version": hashes["classification"]["prompt_version"],
                    "identifier": f"sha256:{hashes['classification']['criteria_sha256']}",
                },
                {
                    "@type": "CreativeWork",
                    "name": "DOME Observatory controlled vocabularies (enrichment prompt)",
                    "url": schema_url,
                    "version": hashes["enrichment"]["prompt_version"],
                    "identifier": f"sha256:{hashes['enrichment']['vocab_sha256']}",
                },
                {
                    "@type": "CreativeWork",
                    "name": "Europe PMC search space",
                    "url": "https://europepmc.org",
                    "identifier": f"sha256:{query_sha256}",
                },
            ],
        },
    }
    if previous:
        dataset["dcat:prev"] = {"@id": release_id(previous)}

    distribution = {
        "@id": EXPORT_DISTRIBUTION_ID,
        "@type": ["dcat:Distribution", "DataDownload"],
        "name": "Whole-corpus export as NDJSON, through the API",
        "dct:title": "Whole-corpus export as NDJSON, through the API",
        "description": "Newline-delimited JSON, one record per line, paged on a cursor "
                       "(X-Next-Cursor); every search filter applies.",
        "contentUrl": f"{ORIGIN}/api/export",
        "dcat:accessURL": {"@id": f"{ORIGIN}/api/export"},
        "encodingFormat": "application/x-ndjson",
        "dct:format": "application/x-ndjson",
        "dct:conformsTo": {"@id": schema_url},
        "dcat:accessService": {"@id": API_ID},
        "license": CC_BY_4,
        "dct:license": {"@id": CC_BY_4},
    }

    service = {
        "@id": API_ID,
        "@type": ["dcat:DataService", "WebAPI"],
        "name": "DOME Observatory API",
        "dct:title": "DOME Observatory API",
        "description": "Read-only REST API: search, single records, whole-corpus export, statistics, "
                       "per-record JSON-LD and OAI-PMH harvesting.",
        "url": API_ID,
        "dcat:endpointURL": {"@id": API_ID},
        "dcat:endpointDescription": {"@id": f"{ORIGIN}/api/docs-json"},
        "documentation": f"{ORIGIN}/download/api",
        "termsOfService": f"{ORIGIN}/download/api",
        "dcat:servesDataset": {"@id": SERIES_ID},
        "provider": {"@id": PUBLISHER_ID},
        "dct:publisher": {"@id": PUBLISHER_ID},
    }

    publisher = {
        "@id": PUBLISHER_ID,
        "@type": ["foaf:Organization", "Organization"],
        "name": "BioComputingUP, University of Padua",
        "foaf:name": "BioComputingUP, University of Padua",
        "url": PUBLISHER_ID,
        "parentOrganization": {"@id": UNIVERSITY_ROR, "@type": "Organization", "name": "University of Padua"},
    }

    contact = {
        "@id": CONTACT_ID,
        "@type": ["vcard:Organization", "ContactPoint"],
        "vcard:fn": "DOME Observatory",
        "vcard:hasEmail": {"@id": f"mailto:{CONTACT_EMAIL}"},
        "name": "DOME Observatory",
        "email": CONTACT_EMAIL,
        "contactType": "dataset enquiries",
    }

    people_nodes = [p for p in people if "@id" in p]
    return {"@context": CONTEXT,
            "@graph": [catalog, series, dataset, distribution, service, publisher, contact, *people_nodes]}


def write_release(document: dict[str, Any], observatory_dir: Path, release: str, overwrite: bool) -> Path:
    """Writes the month's file, and moves metadata/CURRENT to it unless CURRENT already names a later
    month (a backfill must not wind the published catalogue back)."""
    metadata = observatory_dir / "metadata"
    target = metadata / "releases" / release / "dataset.jsonld"
    if target.exists() and not overwrite:
        raise SystemExit(f"{target} exists -- a published release is immutable. Pass --overwrite only "
                         f"if that month has not been committed in dome-ml-observatory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    current = metadata / "CURRENT"
    if not current.exists() or current.read_text(encoding="utf-8").strip() <= release:
        current.write_text(release + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=None, help="Default: the latest verify_corpus report.")
    parser.add_argument("--release", default=None, help="YYYY-MM. Default: the month the report was taken.")
    parser.add_argument("--observatory-dir", type=Path,
                        default=Path(os.environ.get("DOME_OBSERVATORY_DIR", REPO.parent / "dome-ml-observatory")))
    parser.add_argument("--write", action="store_true", help="Write into the sister repository.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a month not yet committed there.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Record the commit even though this working tree has uncommitted changes.")
    args = parser.parse_args()

    schema_version = authored_schema_version()
    report_path = args.report or latest_report()
    report = load_report(report_path, schema_version)
    release = args.release or report["finished_at"][:7]
    commit, dirty = git_commit()
    if dirty and args.write and not args.allow_dirty:
        raise SystemExit("this working tree has uncommitted changes, so the recorded commit would not be "
                         "the code that ran -- commit first, or pass --allow-dirty")
    document = build_document(
        report,
        release=release,
        schema_version=schema_version,
        hashes=json.loads(PROMPT_HASHES.read_text(encoding="utf-8")),
        query_sha256=SearchSpace.load().sha256(),
        commit=commit,
        people=creators(),
        previous=previous_release(args.observatory_dir, release),
    )
    print(f"report {report_path.name}, release {release}, schema {schema_version}, commit {commit[:12]}"
          + (" (dirty)" if dirty else ""))
    if not args.write:
        print(json.dumps(document, indent=2, ensure_ascii=False))
        print("\nDRY RUN -- re-run with --write to publish it into dome-ml-observatory.")
        return
    target = write_release(document, args.observatory_dir, release, args.overwrite)
    print(f"wrote {target}")
    print("next, in dome-ml-observatory: check /api/catalog serves it, then commit metadata/.")


if __name__ == "__main__":
    main()
