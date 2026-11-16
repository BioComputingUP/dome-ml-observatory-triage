"""Finds `ontology_mappings` for the seed terms e3 adds, by the method in docs/vocabulary_ontology_mappings.md:
EBI OLS4 exact search on label and synonym as the candidate source, an ancestor-based homonym guard, NLM
confirmation of every MeSH id, and EBI Zooma (HIGH confidence) as corroboration.

Accepted automatically, as in v1.5.1:
- `skos:exactMatch`: the ontology's label agrees with the canonical (plural, comma inversion, case, spacing);
- `skos:closeMatch`: the label agrees with an alias that names the term itself.
Never adopted: acronyms of five characters or fewer, hits reached only through a synonym, MeSH ids NLM does
not confirm, hits failing the homonym guard, ties within one ontology. At most one entry per ontology, in the
order mesh, aio, ncit, obi, swo, stato, edam; `[]` when nothing is accepted.

    docker compose run --rm tools python scripts/e3_ontology_mappings.py --terms "AlphaFold" "VGG" ...
Prints one JSON object {canonical: [mapping, ...]} to stdout. Network: EBI OLS4, EBI Zooma, NLM (all public).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from datetime import date

import requests

OLS = "https://www.ebi.ac.uk/ols4/api"
ZOOMA = "https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate"
NLM = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
ONTOLOGIES = ("mesh", "aio", "ncit", "obi", "swo", "stato", "edam")
GUARD = {
    "mesh": r"machine learning|artificial intelligence|algorithm|neural network|statistic|comput|model",
    "ncit": r"machine learning|artificial intelligence|algorithm|neural network|statistic|comput|model",
    "obi": r"data transformation|algorithm|planned process",
    "swo": r"algorithm|software",
    "edam": r"operation|topic",
    "aio": r".",      # AIO is an AI ontology; every term is in scope
    "stato": r"model|statistic|test|regression",
}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "dome-ml-observatory-triage vocabulary mapping"


def norm(s: str) -> str:
    s = s.lower().replace("-", " ").replace("_", " ")
    if "," in s:  # "Neural Networks, Computer" -> "computer neural networks"
        head, _, tail = s.partition(",")
        s = f"{tail.strip()} {head.strip()}"
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(?<=\w)s\b", "", s)  # crude plural fold, applied to both sides


def _get(url: str, **params) -> dict | list | None:
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    return None


def ols_candidates(query: str) -> list[dict]:
    data = _get(f"{OLS}/search", q=query, ontology=",".join(ONTOLOGIES), exact="true",
                queryFields="label,synonym", rows=50, fieldList="iri,label,ontology_name,obo_id,short_form,type")
    return [d for d in ((data or {}).get("response") or {}).get("docs", []) if d.get("type") == "class"]


def ancestors_text(doc: dict) -> str:
    iri = urllib.parse.quote(urllib.parse.quote(doc["iri"], safe=""), safe="")
    data = _get(f"{OLS}/ontologies/{doc['ontology_name']}/terms/{iri}/hierarchicalAncestors", size=200)
    terms = ((data or {}).get("_embedded") or {}).get("terms", [])
    return " | ".join(t.get("label", "") for t in terms).lower()


def nlm_confirms(label: str, mesh_id: str) -> bool:
    data = _get(NLM, label=label, match="exact", limit=5) or []
    return any(str(d.get("resource", "")).endswith("/" + mesh_id) for d in data)


def zooma_tags(query: str) -> set[str]:
    data = _get(ZOOMA, propertyValue=query) or []
    return {t for a in data if a.get("confidence") == "HIGH" for t in a.get("semanticTags", [])}


def curie(doc: dict) -> str:
    onto = doc["ontology_name"]
    if onto == "mesh":
        return f"mesh:{doc['short_form']}"
    return doc.get("obo_id") or f"{onto.upper()}:{doc['short_form']}"


def map_term(canonical: str, aliases: list[str], today: str) -> list[dict]:
    queries = [("canonical", canonical)] + [("alias", a) for a in aliases if len(a.replace(" ", "")) > 5]
    accepted: dict[str, dict] = {}
    for matched_on, q in queries:
        for doc in ols_candidates(q):
            onto = doc["ontology_name"]
            if onto in accepted or norm(doc.get("label", "")) != norm(q):
                continue  # synonym-only hits are never adopted
            if not re.search(GUARD[onto], ancestors_text(doc) or doc.get("label", "").lower()):
                continue
            if onto == "mesh" and not nlm_confirms(doc["label"], doc["short_form"]):
                continue
            same = [d for d in ols_candidates(q) if d["ontology_name"] == onto and norm(d.get("label", "")) == norm(q)]
            if len({d["iri"] for d in same}) > 1:
                continue  # a tie within one ontology
            sources = ["ols4"] + (["zooma"] if doc["iri"] in zooma_tags(q) else []) + (["nlm"] if onto == "mesh" else [])
            accepted[onto] = {
                "ontology": onto, "id": curie(doc), "iri": doc["iri"], "label": doc["label"],
                "predicate": "skos:exactMatch" if matched_on == "canonical" else "skos:closeMatch",
                "matched_on": matched_on, "match_string": q, "source": "+".join(sources), "checked": today,
            }
    return [accepted[o] for o in ONTOLOGIES if o in accepted]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True, help="the e3 model_type_seed_vocab.json (reads canonical + aliases)")
    ap.add_argument("--terms", nargs="+", required=True, help="canonical names to map")
    args = ap.parse_args(argv)
    seed = {t["canonical"]: t for t in json.load(open(args.seed, encoding="utf-8"))["terms"]}
    today = date.today().isoformat()
    out = {}
    for name in args.terms:
        out[name] = map_term(name, seed[name].get("aliases", []), today)
        print(f"{name}: {[m['id'] for m in out[name]]}", file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
