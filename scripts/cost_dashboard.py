#!/usr/bin/env python3
"""Writes COST_DASHBOARD.md: what the next classification and enrichment runs cost.

Runs are costed at the **billed** rate: the balance deltas logged in
data/processed/cost_estimates/deepseek_real_cost_log.csv, and for enrichment never below
`planning_usd_per_1k` in pricing/token_profiles.yaml. Beside that, a list-price model (tokens per
record from pricing/token_profiles.yaml x prices from pricing/pricing.yaml) compares providers and
peak against off-peak; it is not billed, and for enrichment it undershoots the bill. Corpus counts
are live from moros with --live, else pricing/corpus_snapshot.json.

    python3 scripts/cost_dashboard.py --live --balance     # read-only moros query + DeepSeek balance
    python3 scripts/cost_dashboard.py                      # from the last snapshot
    python3 scripts/cost_dashboard.py --events-classification <csv> --events-enrichment <csv>
                                                           # re-measure token profiles from event logs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PRICING = REPO / "pricing" / "pricing.yaml"
PROFILES = REPO / "pricing" / "token_profiles.yaml"
SNAPSHOT = REPO / "pricing" / "corpus_snapshot.json"
REAL_COST_LOG = REPO / "data" / "processed" / "cost_estimates" / "deepseek_real_cost_log.csv"
OUT = REPO / "COST_DASHBOARD.md"

FLASH = "deepseek-flash"          # pricing.yaml key for the model the `flash` tier is answered by
CLASSIFY_RECORDS_PER_MIN = 4500   # 13,499 records in ~3 min at --concurrency 400 (2026-09-03)


# ----------------------------------------------------------------------------- inputs

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def live_counts() -> dict:
    sys.path.insert(0, str(REPO / "moros_pipeline" / "scripts"))
    from moros_client import Moros  # noqa: E402
    max_ms = 180_000
    with Moros.from_env() as moros:
        col = moros.collection
        pos = {"llm_classification.classification": "positive"}
        not_enriched = {"llm_enrichment.batch_id": None}
        has_abstract = {"publication_metadata.abstract": {"$nin": [None, ""]}}
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        counts = {
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": col.estimated_document_count(),
            "positive": col.count_documents(pos, maxTimeMS=max_ms),
            "negative": col.count_documents({"llm_classification.classification": "negative"}, maxTimeMS=max_ms),
            "undeterminable": col.count_documents({"llm_classification.classification": "undeterminable"}, maxTimeMS=max_ms),
            "enriched": col.count_documents({"llm_enrichment.batch_id": {"$ne": None}}, maxTimeMS=max_ms),
            "positive_not_enriched": col.count_documents({**pos, **not_enriched}, maxTimeMS=max_ms),
            "positive_not_enriched_with_abstract": col.count_documents({**pos, **not_enriched, **has_abstract}, maxTimeMS=max_ms),
            "citations_never_fetched": col.count_documents({"publication_metadata.citation_count_updated": None}, maxTimeMS=max_ms),
            "citations_older_than_30d": col.count_documents({"publication_metadata.citation_count_updated": {"$lt": cutoff}}, maxTimeMS=max_ms),
        }
    return counts


def deepseek_balance() -> dict | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        return None
    try:
        out = subprocess.run(
            ["curl", "-s", "--max-time", "20", "-H", f"Authorization: Bearer {key}",
             "https://api.deepseek.com/user/balance"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        data = json.loads(out)
        info = data["balance_infos"][0]
        return {"currency": info["currency"], "total_balance": float(info["total_balance"]),
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    except Exception as exc:  # network, key, shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def billed_runs(path: Path = REAL_COST_LOG) -> list[dict]:
    """Every billed run in the real cost log, in the order it was logged. A row whose mode starts
    with `enrich` is enrichment; everything else is classification."""
    if not path.exists():
        return []
    runs = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                n = int(row["n_records"])
                usd = float(row["real_usd"])
            except (KeyError, TypeError, ValueError):
                continue
            if n <= 0:
                continue
            mode = row.get("mode") or ""
            runs.append({
                "date": row.get("date") or "", "step": "enrichment" if mode.startswith("enrich") else "classification",
                "tier": row.get("tier") or "", "mode": mode, "n": n, "usd": usd, "per_record": usd / n,
                "source": row.get("source") or "",
            })
    return runs


def measure_profile(path: Path, cols: list[str]) -> dict:
    sums = {c: 0.0 for c in cols}
    n = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            for c in cols:
                try:
                    sums[c] += float(row.get(c) or 0)
                except ValueError:
                    pass
    return {"n_events": n, **{c: round(sums[c] / max(n, 1)) for c in cols}}


# ----------------------------------------------------------------------------- model

def per_record_usd(price: dict, hit: float, miss: float, out: float) -> float:
    return (hit * price["input_cache_hit"] + miss * price["input_cache_miss"] + out * price["output"]) / 1e6


def planning_rates(profiles: dict, runs: list[dict]) -> dict:
    """USD per record every run is costed at. Classification: the latest billed classification run.
    Enrichment: the latest billed enrichment run, but never below `planning_usd_per_1k` -- lowering
    that floor is a decision for a person, not something a cheap run does on its own."""
    c, e = profiles["classification"], profiles["enrichment"]
    cls_runs = [r for r in runs if r["step"] == "classification"]
    enr_runs = [r for r in runs if r["step"] == "enrichment"]
    if cls_runs:
        cls, cls_src = cls_runs[-1]["per_record"], f"Billed: {cls_runs[-1]['n']:,} records on {cls_runs[-1]['date']}"
    else:
        cls, cls_src = c["real_anchor_usd_per_record"], c.get("real_anchor_source", "token_profiles.yaml")
    floor = float(e.get("planning_usd_per_1k", 10.0)) / 1000
    if enr_runs and enr_runs[-1]["per_record"] >= floor:
        enr, enr_src = enr_runs[-1]["per_record"], f"Billed: {enr_runs[-1]['n']:,} records on {enr_runs[-1]['date']}."
    else:
        enr, enr_src = floor, e.get("planning_source", "token_profiles.yaml planning_usd_per_1k")
        if enr_runs:
            enr_src += (f" The latest billed run came to ${enr_runs[-1]['per_record'] * 1000:.2f} per 1,000 "
                        f"({enr_runs[-1]['n']:,} records, {enr_runs[-1]['date']}); the higher figure is kept.")
    return {"classification": cls, "classification_source": cls_src, "enrichment": enr, "enrichment_source": enr_src}


def model_label(model: dict, fallback: str) -> str:
    return (model.get("version") or fallback).replace("-", " ")


def scenarios(pricing: dict) -> list[tuple[str, dict]]:
    ds = pricing["deepseek"]["models"][FLASH]
    glm = pricing["zai"]["models"]["glm-5.3-flash"]
    name = model_label(ds, FLASH)
    rows = [
        (f"{name}, off-peak", ds["off_peak"]),
        (f"{name}, peak", ds["peak"]),
        ("GLM-5.3-Flash, list", glm["list"]),
    ]
    promo = glm.get("promo")
    if promo:
        ends = datetime.fromisoformat(promo["ends"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) < ends:
            rows.append((f"GLM-5.3-Flash, promo (until {ends:%Y-%m-%d} UTC)", promo))
    return rows


def fmt_usd(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


def hours(records: int, per_min: float) -> str:
    h = records / per_min / 60
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


# ----------------------------------------------------------------------------- render

def render(pricing: dict, profiles: dict, counts: dict, balance: dict | None, runs: list[dict]) -> str:
    c = profiles["classification"]
    e = profiles["enrichment"]
    cls_tokens = (c["input_tokens"] - c["record_tokens_estimate"], c["record_tokens_estimate"], c["output_tokens"])
    enr_tokens = (e["cache_hit_tokens"], e["input_tokens"] - e["cache_hit_tokens"], e["output_tokens"])
    scen = scenarios(pricing)
    rates = planning_rates(profiles, runs)
    ds = pricing["deepseek"]["models"][FLASH]
    enr_list_offpeak = per_record_usd(ds["off_peak"], *enr_tokens)

    last_batch = c.get("last_incremental_batch_records", 13499)
    last_batch_pos = c.get("last_incremental_batch_positives", 7369)
    backlog = counts["positive_not_enriched_with_abstract"]
    enr_per_min = e["records_per_min_at_concurrency_800"]

    lines = []
    lines.append("# Cost dashboard")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · prices as of **{pricing['as_of']}** "
                 f"(`pricing/pricing.yaml`) · corpus counts as of **{counts['fetched_at'][:16].replace('T', ' ')} UTC** "
                 f"· token profiles measured **{profiles['measured_on']}** on {profiles.get('model_version', profiles.get('model_id'))} "
                 f"(`pricing/token_profiles.yaml`) · billed runs from `data/processed/cost_estimates/deepseek_real_cost_log.csv`.")
    lines.append("Regenerate: `python3 scripts/cost_dashboard.py --live --balance`. Timing: `python3 scripts/offpeak_window.py --minutes N`.")
    lines.append("")

    # Corpus
    lines.append("## Corpus now")
    lines.append("")
    lines.append("| | Records |")
    lines.append("|---|---:|")
    lines.append(f"| Documents in `dome_observatory.Content` | {counts['total']:,} |")
    lines.append(f"| Positive / negative / undeterminable | {counts['positive']:,} / {counts['negative']:,} / {counts['undeterminable']:,} |")
    lines.append(f"| Enriched | {counts['enriched']:,} |")
    lines.append(f"| **Positives left to enrich** (with an abstract) | **{backlog:,}** ({counts['positive_not_enriched']:,} incl. no-abstract) |")
    lines.append(f"| Citation counts never fetched / older than 30 days | {counts['citations_never_fetched']:,} / {counts['citations_older_than_30d']:,} |")
    lines.append("")

    # Balance
    lines.append("## DeepSeek balance")
    lines.append("")
    if balance and "total_balance" in balance:
        lines.append(f"**{balance['currency']} {balance['total_balance']:.2f}** at {balance['checked_at'][:16].replace('T', ' ')} UTC "
                     f"(`GET /user/balance`). Top up before any run whose planning cost below exceeds it.")
    elif balance and "error" in balance:
        lines.append(f"Not read: {balance['error']}")
    else:
        lines.append("Not read (run with `--balance` and `DEEPSEEK_API_KEY` set).")
    lines.append("")

    # What a record costs
    lines.append("## What a record costs")
    lines.append("")
    lines.append("Every run below is costed at the **planning rate**, which comes from the bill, not from list prices:")
    lines.append("")
    lines.append(f"- Classification: **${rates['classification']:.6f} per record** "
                 f"(${rates['classification'] * 1000:.2f} per 1,000). {rates['classification_source']}.")
    lines.append(f"- Enrichment: **${rates['enrichment'] * 1000:.2f} per 1,000 records** "
                 f"(${rates['enrichment']:.4f} per record). {rates['enrichment_source']}")
    lines.append("")
    lines.append("### Billed runs (balance deltas)")
    lines.append("")
    if runs:
        lines.append("| Date | Step | Tier, mode | Records | Billed | Per 1,000 |")
        lines.append("|---|---|---|---:|---:|---:|")
        for r in runs:
            lines.append(f"| {r['date']} | {r['step']} | {r['tier']}, {r['mode']} | {r['n']:,} | ${r['usd']:.2f} | ${r['per_record'] * 1000:.2f} |")
    else:
        lines.append("None logged yet.")
    lines.append("")
    lines.append("### List-price model (tokens × list price, not billed)")
    lines.append("")
    lines.append(f"For comparing providers and peak against off-peak only. For enrichment it gives {fmt_usd(1000 * enr_list_offpeak)} "
                 f"per 1,000 off-peak against a planning rate of ${rates['enrichment'] * 1000:.2f}: it undershoots the bill, "
                 "so never budget from it.")
    lines.append("")
    lines.append("| Step | Tokens per record (cache hit / miss / output) | " + " | ".join(n for n, _ in scen) + " |")
    lines.append("|---|---|" + "---:|" * len(scen))
    cls_cells = " | ".join(fmt_usd(per_record_usd(p, *cls_tokens)) for _, p in scen)
    lines.append(f"| Classification (prompt {c['prompt_version']}) | {cls_tokens[0]:,} / {cls_tokens[1]:,} / {cls_tokens[2]:,} | {cls_cells} |")
    enr_cells = " | ".join(fmt_usd(per_record_usd(p, *enr_tokens)) for _, p in scen)
    lines.append(f"| Enrichment (prompt {e['prompt_version']}) | {enr_tokens[0]:,} / {enr_tokens[1]:,} / {enr_tokens[2]:,} "
                 f"({e['reasoning_tokens']:,} reasoning) | {enr_cells} |")
    lines.append("")
    lines.append("Per 1,000 records:")
    lines.append("")
    lines.append("| Step | " + " | ".join(n for n, _ in scen) + " |")
    lines.append("|---|" + "---:|" * len(scen))
    lines.append("| Classification | " + " | ".join(fmt_usd(1000 * per_record_usd(p, *cls_tokens)) for _, p in scen) + " |")
    lines.append("| Enrichment | " + " | ".join(fmt_usd(1000 * per_record_usd(p, *enr_tokens)) for _, p in scen) + " |")
    lines.append("")

    # Workloads
    lines.append("## What the next runs cost")
    lines.append("")
    lines.append("| Run | Records | **Planning (billed rate)** | " + " | ".join(f"{n} (list model)" for n, _ in scen) + " | Wall time |")
    lines.append("|---|---:|---:|" + "---:|" * len(scen) + "---|")
    work = [
        (f"Classify one incremental batch (last batch: {last_batch:,})", last_batch, "classification", cls_tokens, CLASSIFY_RECORDS_PER_MIN, 400),
        (f"Enrich that batch's positives (last batch: {last_batch_pos:,})", last_batch_pos, "enrichment", enr_tokens, enr_per_min, 800),
        ("Enrich 300 positives (a capped cohort)", 300, "enrichment", enr_tokens, enr_per_min, 800),
        ("Enrich 10,000 positives (one journal-sized cohort)", 10_000, "enrichment", enr_tokens, enr_per_min, 800),
        (f"**Enrich every remaining positive** ({backlog:,})", backlog, "enrichment", enr_tokens, enr_per_min, 800),
    ]
    for label, n, step, toks, rate, conc in work:
        cells = " | ".join(fmt_usd(n * per_record_usd(p, *toks)) for _, p in scen)
        lines.append(f"| {label} | {n:,} | **{fmt_usd(n * rates[step])}** | {cells} | {hours(n, rate)} at concurrency {conc} |")
    lines.append("")
    lines.append(f"At the planning rate the full enrichment backlog is **{fmt_usd(backlog * rates['enrichment'])}**; the "
                 f"list-price model says {fmt_usd(backlog * enr_list_offpeak)} off-peak. Classification of a monthly batch "
                 "is a rounding error next to it.")
    lines.append("")

    # Timing
    pw = pricing["deepseek"]["peak_windows_utc"]
    lines.append("## Best time to run (DeepSeek)")
    lines.append("")
    lines.append("Peak, at double price: " + " and ".join(f"{w['from']}–{w['to']} UTC" for w in pw) + f", {pw[0]['days']}. "
                 "Everything else, including all weekend, is off-peak.")
    lines.append("")
    lines.append("- A monthly classification batch takes minutes: run it any off-peak hour.")
    lines.append(f"- A 10,000-record enrichment takes ~{hours(10_000, enr_per_min)}: "
                 "start after 10:00 UTC on a weekday, or any time at the weekend.")
    lines.append(f"- The full backlog takes ~{hours(backlog, enr_per_min)} at concurrency 800: "
                 "only a weekend (Fri 10:00 UTC → Mon 01:00 UTC) holds it entirely off-peak; otherwise run it as "
                 "journal cohorts, each inside one off-peak stretch. Resumability makes splitting free.")
    lines.append("- GLM-5.3-Flash publishes no off-peak rate; its column is the same price at any hour.")
    lines.append("")

    # Assumptions
    lines.append("## Assumptions and caveats")
    lines.append("")
    measured_on = profiles.get("model_version")
    if measured_on and ds.get("version") and measured_on != ds["version"]:
        lines.append(f"- **The model changed.** Token profiles were measured on {measured_on}; DeepSeek now answers "
                     f"`{profiles.get('model_id')}` with {ds['version']} (from {str(ds.get('effective_from', ''))[:10]}), "
                     f"and the list prices above are {ds['version']}'s. Its token use is unmeasured until its first events "
                     "files are re-measured (`--events-classification` / `--events-enrichment`), and whether it agrees with "
                     "the validated model is a separate check (see ROADMAP.md).")
    lines.append(f"- **Enrichment's list-price model undershoots its bill.** Tokens × list price gave "
                 f"${float(e.get('token_model_usd_per_1k', 0)):.2f} per 1,000 on V4-Flash; enrichment is billed at about "
                 f"${float(e.get('planning_usd_per_1k', 10.0)):.0f}. Budget from billed runs only: bracket every paid run with "
                 "`GET /user/balance` reads and append the delta to the real cost log.")
    lines.append("- The GLM columns apply DeepSeek's token counts to Z.ai's prices: a different tokenizer and a "
                 "different reasoning budget would change them, and **GLM-5.3-Flash has not been validated** against "
                 "the human benchmark or the enrichment agreement check. They are a price comparison, not an approved "
                 "substitute (see ROADMAP.md).")
    lines.append("- Classification cache hits are modelled (the event log does not record them): everything but the "
                 f"~{c['record_tokens_estimate']} per-record tokens is treated as a prefix-cache hit. On V4-Flash that "
                 f"model came within ~6% of the billed ${c['real_anchor_usd_per_record']:.6f}/record.")
    lines.append("- Enrichment cost is not reducible by settings: lower reasoning effort cost more with six times the "
                 "vocabulary violations; thinking off was 88% cheaper and agreed with production on all six fields "
                 "for 0% of records.")
    lines.append(f"- About {c['no_abstract_share'] * 100:.0f}% of new records have no abstract and are never sent to the model; "
                 "the backlog figure already excludes abstract-less positives.")
    lines.append("- List prices drift. The `cost-estimate` skill re-reads both pricing pages before any paid run and "
                 "updates `pricing/pricing.yaml`; real spend is confirmed from the balance delta afterwards.")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Read corpus counts from moros (read-only) and update the snapshot.")
    parser.add_argument("--balance", action="store_true", help="Read the DeepSeek balance.")
    parser.add_argument("--events-classification", type=Path, default=None)
    parser.add_argument("--events-enrichment", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    pricing = load_yaml(PRICING)
    profiles = load_yaml(PROFILES)

    if args.events_classification or args.events_enrichment:
        if args.events_classification:
            m = measure_profile(args.events_classification, ["input_tokens", "output_tokens"])
            profiles["classification"].update(m)
            profiles["classification"]["source"] = f"{args.events_classification.name}, n={m['n_events']:,}"
        if args.events_enrichment:
            m = measure_profile(args.events_enrichment, ["input_tokens", "cache_hit_tokens", "output_tokens", "reasoning_tokens"])
            profiles["enrichment"].update(m)
            profiles["enrichment"]["source"] = f"{args.events_enrichment.name}, n={m['n_events']:,}"
        profiles["measured_on"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        PROFILES.write_text(yaml.safe_dump(profiles, sort_keys=False), encoding="utf-8")
        print(f"rewrote {PROFILES}")

    if args.live:
        counts = live_counts()
        SNAPSHOT.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {SNAPSHOT}")
    elif SNAPSHOT.exists():
        counts = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    else:
        raise SystemExit("no pricing/corpus_snapshot.json yet -- run with --live (needs moros_pipeline/.env and the VPN)")

    balance = deepseek_balance() if args.balance else None
    args.out.write_text(render(pricing, profiles, counts, balance, billed_runs()), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
