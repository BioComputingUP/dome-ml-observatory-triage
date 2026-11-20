"""Field-by-field agreement between two enrichment events files over the same records -- the check that a new
provider or prompt version reproduces a validated run (docs/enrichment_ebi.md).

For each tag field: exact agreement (the same set of values) and mean Jaccard overlap, over the records
both runs parsed. The headline split is records where the candidate used none of the terms its prompt
added (where the two should agree as closely as two runs of the reference do) versus records where it
used one (where a difference is the point of the new prompt). Also: parse errors, truncations, output
tokens per record.

    docker compose run --rm tools python scripts/compare_enrichment_runs.py \\
        --reference ../dome-ml-observatory-research/dome-ml-observatory-glm53-validation/runs/e2/events_wave2.csv \\
        --candidate moros_pipeline/output/enrichment_e3/qc_wave2/input_wave2_250_events.csv \\
        --new-terms-from curation_criteria/e3/model_type_seed_vocab.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

csv.field_size_limit(sys.maxsize)

FIELDS = ("domain_tier1", "domain_tier2", "domain_tier3", "model_use", "learning_paradigm", "model_family",
          "model_type")
MODELLING = ("learning_paradigm", "model_family", "model_type")


def load(path: Path) -> dict[str, dict]:
    """The last event per record (append-only logs), parse errors kept so they can be counted."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = {r["record_id"]: r for r in csv.DictReader(f)}
    return rows


def values(row: dict, field: str) -> frozenset:
    try:
        return frozenset(v.lower() for v in json.loads(row.get(field) or "[]"))
    except json.JSONDecodeError:
        return frozenset()


def jaccard(a: frozenset, b: frozenset) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def new_terms(seed_path: Path) -> set[str]:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    return {t["canonical"].lower() for t in seed["terms"] if "added" in (t.get("e3") or {})}


def summarise(ref: dict, cand: dict, ids: list[str]) -> dict:
    out = {"n": len(ids)}
    for f in FIELDS:
        exact = [values(ref[i], f) == values(cand[i], f) for i in ids]
        jac = [jaccard(values(ref[i], f), values(cand[i], f)) for i in ids]
        out[f] = (mean(exact) if ids else 0.0, mean(jac) if ids else 0.0)
    out["all_modelling"] = mean(all(values(ref[i], f) == values(cand[i], f) for f in MODELLING) for i in ids) if ids else 0.0
    return out


def stats(rows: dict) -> str:
    n = len(rows)
    pe = sum(r.get("parse_status") == "parse_error" for r in rows.values())
    tr = sum(r.get("finish_reason") == "length" for r in rows.values())
    toks = [float(r.get("output_tokens") or 0) for r in rows.values() if r.get("parse_status") != "parse_error"]
    return f"{n} records, {pe} parse errors, {tr} truncated, {mean(toks) if toks else 0:,.0f} output tokens/record"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--new-terms-from", type=Path, help="seed vocab whose entries marked e3.added are 'new'")
    args = ap.parse_args(argv)
    ref, cand = load(args.reference), load(args.candidate)
    ok = lambda r: r.get("parse_status") != "parse_error"  # noqa: E731
    both = sorted(i for i in set(ref) & set(cand) if ok(ref[i]) and ok(cand[i]))
    added = new_terms(args.new_terms_from) if args.new_terms_from else set()
    uses_new = [i for i in both if values(cand[i], "model_type") & added]
    without = [i for i in both if i not in set(uses_new)]

    print(f"reference: {args.reference.name}: {stats(ref)}")
    print(f"candidate: {args.candidate.name}: {stats(cand)}")
    print(f"compared: {len(both)} records parsed by both; {len(uses_new)} where the candidate used a new term\n")
    groups = [("all", both), ("no new term", without), ("uses a new term", uses_new)]
    sums = [(name, summarise(ref, cand, ids)) for name, ids in groups]
    print(f"{'field':<20}" + "".join(f"{name + f' (n={s['n']})':>30}" for name, s in sums))
    print(f"{'':<20}" + "".join(f"{'exact / jaccard':>30}" for _ in sums))
    for f in FIELDS:
        print(f"{f:<20}" + "".join(f"{f'{s[f][0]:.1%} / {s[f][1]:.2f}':>30}" for _, s in sums))
    print(f"{'all 3 modelling':<20}" + "".join(f"{f'{s['all_modelling']:.1%}':>30}" for _, s in sums))
    return 0


if __name__ == "__main__":
    sys.exit(main())
