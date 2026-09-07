"""Step 20j's enrichment success/usage charts -- how the tagging actually went, per field, over
the real event log. Same pure-`_data()`-helper + thin-`plot_()`-renderer convention as
`dataset_profile.py`/`agreement.py`, reusing their shared `_mpl`/`_add_caption` helpers.

Every list-valued field in `enrichment_classification_events.csv` is JSON-encoded (matching the
`mesh_headings` convention); everything here decodes with a tolerant `_loads` that treats a
missing/broken cell as an empty list rather than crashing a whole profile over one bad row.
"""

from __future__ import annotations

import json

import pandas as pd

from dome_triage.llm_classify.enrichment import LIST_FIELDS, PARSE_ERROR
from dome_triage.reporting.dataset_profile import _add_caption, _mpl

PROFILE_README = """# Enrichment Profile (Step 20j)

How the positive-set enrichment tagging actually went -- success, coverage, and consistency, per
field, over the real `enrichment_classification_events.csv`.

## Charts

- `tag_count_distribution.png` -- for each of the six fields, how many records got 0/1/2/3+ tags.
  A large 0-tag bar on `learning_paradigm` would mean the model often could not identify a
  paradigm at all; the domain tiers legitimately allow 0 ("not applicable").
- `top_model_types.png` -- most frequent model_type values. Because seed-matching normalizes
  spelling at parse time, near-duplicates here ("XGBoost" vs "extreme gradient boosting") indicate
  a seed-vocabulary gap worth closing before the full-landscape pass (Step 23b).
- `paradigm_family_distribution.png` -- learning_paradigm and model_family tag frequencies.
- `normalization_and_violations.png` -- what share of model_type values came out seed-canonical vs
  novel free-text, and the per-field vocabulary-violation rate (unknown term / cap exceeded --
  tolerated and logged at parse time, never silently dropped).

## Reading it

The trial's "did it work" question is answered by, together: a near-zero parse_error count, a low
violation rate, high seed-canonical share on model_type, and tag-count distributions that look
like real papers (mostly 1 paradigm, 1-2 families, 1-3 model types).
"""


def _loads(cell) -> list:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        value = json.loads(cell)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def latest_ok_events(events: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Latest non-parse-error event per record for this tier -- same last-event-wins convention as
    everywhere else in this project."""
    subset = events[(events["model_tier"] == tier) & (events["parse_status"] != PARSE_ERROR)]
    if subset.empty:
        return subset
    return subset.sort_values("timestamp").groupby("record_id").last().reset_index()


def tag_count_distribution(ok_events: pd.DataFrame) -> pd.DataFrame:
    """Rows: field; columns: '0','1','2','3+' -- count of records with that many tags."""
    rows = {}
    for field in LIST_FIELDS:
        counts = ok_events[field].map(lambda cell: len(_loads(cell)))
        bucketed = counts.map(lambda n: "3+" if n >= 3 else str(n))
        rows[field] = bucketed.value_counts().reindex(["0", "1", "2", "3+"], fill_value=0)
    return pd.DataFrame(rows).T


def value_frequencies(ok_events: pd.DataFrame, field: str, top_n: int | None = None) -> pd.Series:
    exploded = ok_events[field].map(_loads).explode().dropna()
    counts = exploded.value_counts()
    return counts.head(top_n) if top_n else counts


def normalization_breakdown(ok_events: pd.DataFrame, seed_vocab: dict) -> dict:
    """model_type values: seed-canonical (normalization worked or the model used the canonical
    name) vs novel free-text (a genuinely unlisted method -- expected and allowed)."""
    canonical = {entry["canonical"] for entry in seed_vocab["terms"]}
    values = ok_events["model_type"].map(_loads).explode().dropna()
    n_canonical = int(values.isin(canonical).sum())
    n_novel = int(len(values) - n_canonical)
    return {"n_model_type_values": int(len(values)), "n_seed_canonical": n_canonical, "n_novel_free_text": n_novel}


def violation_stats(ok_events: pd.DataFrame) -> dict:
    per_field: dict[str, int] = {}
    n_events_with_violation = 0
    for cell in ok_events["vocab_violations"]:
        violations = _loads(cell)
        if violations:
            n_events_with_violation += 1
        for violation in violations:
            field = str(violation).split(":", 1)[0]
            per_field[field] = per_field.get(field, 0) + 1
    return {"n_events_with_violation": n_events_with_violation, "violations_per_field": per_field}


def build_profile(events: pd.DataFrame, vocabs: dict, tier: str, output_dir) -> tuple[dict, list]:
    """Renders all four charts + computes the metadata dict. Returns (metadata, output_paths)."""
    ok = latest_ok_events(events, tier)
    tier_events = events[events["model_tier"] == tier]
    n_parse_error = int((tier_events["parse_status"] == PARSE_ERROR).sum())
    plt = _mpl()
    output_paths = []

    # 1. Tag-count distribution, grouped bars per field.
    dist = tag_count_distribution(ok)
    fig, ax = plt.subplots(figsize=(11, 6))
    dist.plot(kind="bar", ax=ax, color=["#8C8C8C", "#55A868", "#4C72B0", "#8172B3"])
    ax.set_ylabel("Records")
    ax.set_title(f"Tags Per Record, Per Field (n={len(ok):,} enriched records, tier={tier})")
    ax.legend(title="Tag count")
    ax.tick_params(axis="x", rotation=30)
    _add_caption(fig, "Domain tiers legitimately allow 0 (not applicable); a large 0 bar on "
                      "learning_paradigm would instead be a real coverage problem.", width=105)
    path = output_dir / "tag_count_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    # 2. Top model types.
    top_types = value_frequencies(ok, "model_type", top_n=25)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top_types) + 1)))
    bars = ax.barh(list(top_types.index[::-1]), top_types.values[::-1], color="#4C72B0")
    for bar, value in zip(bars, top_types.values[::-1]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {int(value):,}", va="center", fontsize=8)
    ax.set_xlabel("Records tagged")
    ax.set_title(f"Top {len(top_types)} Model Types (open vocabulary, seed-normalized)")
    _add_caption(fig, "Near-duplicate spellings appearing separately here indicate a seed-vocabulary "
                      "gap worth closing before the full-landscape pass.", width=105)
    path = output_dir / "top_model_types.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    # 3. Paradigm + family distributions, side by side.
    paradigm = value_frequencies(ok, "learning_paradigm")
    family = value_frequencies(ok, "model_family")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4.5, 0.4 * max(len(paradigm), len(family)) + 1)))
    ax1.barh(list(paradigm.index[::-1]), paradigm.values[::-1], color="#55A868")
    ax1.set_title("learning_paradigm")
    ax2.barh(list(family.index[::-1]), family.values[::-1], color="#8172B3")
    ax2.set_title("model_family")
    for ax in (ax1, ax2):
        ax.set_xlabel("Records tagged")
    fig.suptitle("Modelling-Branch Tag Frequencies (multi-label -- totals exceed record count)")
    path = output_dir / "paradigm_family_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    # 4. Normalization share + violation counts.
    norm = normalization_breakdown(ok, vocabs["model_type_seed"])
    violations = violation_stats(ok)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(["seed-canonical", "novel free-text"], [norm["n_seed_canonical"], norm["n_novel_free_text"]],
            color=["#55A868", "#4C72B0"])
    ax1.set_title(f"model_type Normalization ({norm['n_model_type_values']:,} values)")
    per_field = violations["violations_per_field"] or {"(none)": 0}
    ax2.bar(list(per_field.keys()), list(per_field.values()), color="#C44E52")
    ax2.set_title(f"Vocab Violations ({violations['n_events_with_violation']:,} records affected)")
    ax2.tick_params(axis="x", rotation=30)
    _add_caption(fig, "Violations are tolerated-and-logged at parse time (unknown term or cap "
                      "exceeded), never silently dropped -- this chart is the honest error rate.", width=105)
    path = output_dir / "normalization_and_violations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    n_ok = len(ok)
    metadata = {
        "tier": tier,
        "n_events": int(len(tier_events)),
        "n_enriched_records": n_ok,
        "n_parse_error": n_parse_error,
        "tag_count_distribution": dist.to_dict(),
        "top_model_types": top_types.to_dict(),
        "learning_paradigm_counts": paradigm.to_dict(),
        "model_family_counts": family.to_dict(),
        "normalization": norm,
        "violations": violations,
        "violation_rate": (violations["n_events_with_violation"] / n_ok) if n_ok else 0.0,
    }
    return metadata, output_paths
