"""Step 23a(b): profiles the full landscape classification run (`landscape_classification_events.csv`,
~827k records) -- a real, reproducible version of the manual duplicate/parse-error audit already
done by hand once this data landed, plus a rate comparison against the curated trusted set.

Pure `_*_data()` helpers (testable without matplotlib) wrapped by thin `plot_*()` renderers,
matching `dataset_profile.py`'s convention exactly -- reuses its shared `_mpl`/`_add_caption`/
`_bar_value_labels` helpers rather than duplicating them.

**The curated-vs-landscape comparison is a rate comparison between two DISJOINT populations, not a
per-record agreement/kappa comparison.** `select_bulk_pool_excluding_curated` removed every
already-curated record before the landscape was classified, so there is ~0 record-level overlap by
design -- that is expected, not a bug, and this module says so in the chart caption rather than
computing a kappa that would be meaningless over an empty intersection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dome_triage.reporting.dataset_profile import _add_caption, _bar_value_labels, _mpl

_CLASSIFICATION_ORDER = ["positive", "negative", "undeterminable", "parse_error"]
_CLASSIFICATION_DISPLAY = {
    "positive": "Positive",
    "negative": "Negative",
    "undeterminable": "Undeterminable",
    "parse_error": "Parse Error",
}
_CLASSIFICATION_COLORS = {
    "positive": "#4C72B0",
    "negative": "#C44E52",
    "undeterminable": "#8172B3",
    "parse_error": "#DD8452",
}

_TRUSTED_LABEL_CONFIDENCE = {"human_curated", "registry_confirmed"}


def resolve_latest_per_record(events: pd.DataFrame) -> pd.DataFrame:
    """Last event wins per `record_id` -- the project-wide convention (see `agreement.py`,
    `enrichment_profile.py`, `run_comparison.py`), so a retried `parse_error` resolves to its
    successful second attempt instead of being double-counted."""
    ordered = events.sort_values("timestamp") if "timestamp" in events.columns else events
    return ordered.drop_duplicates(subset=["record_id"], keep="last")


# ---------------------------------------------------------------------------
# 1. Classification breakdown
# ---------------------------------------------------------------------------

def classification_breakdown_data(resolved: pd.DataFrame) -> pd.Series:
    counts = resolved["classification"].value_counts()
    ordered = [c for c in _CLASSIFICATION_ORDER if c in counts.index]
    extra = [c for c in counts.index if c not in _CLASSIFICATION_ORDER]
    return counts.reindex(ordered + extra)


def plot_classification_breakdown(resolved: pd.DataFrame, output_path: Path) -> pd.Series:
    counts = classification_breakdown_data(resolved)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [_CLASSIFICATION_COLORS.get(c, "#8C8C8C") for c in counts.index]
    display = [_CLASSIFICATION_DISPLAY.get(c, c.title()) for c in counts.index]
    bars = ax.bar(display, counts.values, color=colors, edgecolor="white")
    _bar_value_labels(ax, bars, counts.values)
    ax.set_xlabel("Classification")
    ax.set_ylabel("Records")
    ax.set_title("AI/ML Landscape Classification Breakdown")
    n = len(resolved)
    n_unique = resolved["record_id"].nunique()
    _add_caption(
        fig,
        f"Total classified records: {n:,} (unique record_ids: {n_unique:,}). "
        f"flash / primary mode, criteria_sha256 "
        f"{resolved['criteria_sha256'].iloc[0][:12] if 'criteria_sha256' in resolved.columns and n else 'n/a'}...",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# 2. Duplicate / repeat-id audit
# ---------------------------------------------------------------------------

def duplicate_id_audit_data(events: pd.DataFrame) -> dict:
    """Codifies the manual audit already done by hand on this exact data: total rows vs. unique
    ids, the repeat-count distribution, and -- the number that actually matters -- how many of the
    repeats are explained by a `parse_error` being retried (expected, healthy) versus a genuine
    same-identity duplicate receiving the same answer twice (benign, pre-existing in the pool) or a
    real disagreeing repeat (would need investigating)."""
    n_rows = len(events)
    vc = events["record_id"].value_counts()
    n_unique = len(vc)
    repeated = vc[vc > 1]

    repeated_ids = set(repeated.index)
    sub = events[events["record_id"].isin(repeated_ids)]
    has_parse_error = set(sub.loc[sub["classification"] == "parse_error", "record_id"])

    # Classify each repeated id into exactly one bucket: a parse_error that got retried, a genuine
    # duplicate that agreed with itself, or a genuine duplicate that disagreed (nondeterminism).
    n_retried = n_agree = n_disagree = 0
    for record_id, grp in sub.groupby("record_id"):
        if record_id in has_parse_error:
            n_retried += 1
        elif grp["classification"].nunique() == 1:
            n_agree += 1
        else:
            n_disagree += 1

    return {
        "n_rows": n_rows,
        "n_unique_ids": n_unique,
        "n_excess_rows": n_rows - n_unique,
        "n_ids_repeated": len(repeated),
        "repeat_count_distribution": repeated.value_counts().to_dict(),
        "n_repeats_involving_a_parse_error_retry": n_retried,
        "n_repeats_same_answer_both_times": n_agree,
        "n_repeats_genuinely_disagreeing": n_disagree,
    }


def plot_duplicate_id_audit(events: pd.DataFrame, output_path: Path) -> dict:
    data = duplicate_id_audit_data(events)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    rows = [
        ("Total event rows", f"{data['n_rows']:,}"),
        ("Unique record_ids", f"{data['n_unique_ids']:,}"),
        ("Excess rows (rows - unique ids)", f"{data['n_excess_rows']:,}"),
        ("record_ids appearing more than once", f"{data['n_ids_repeated']:,}"),
        ("  ...involving a retried parse_error", f"{data['n_repeats_involving_a_parse_error_retry']:,}"),
        ("  ...same answer both times (genuine pool duplicate)", f"{data['n_repeats_same_answer_both_times']:,}"),
        ("  ...genuinely disagreeing (model nondeterminism)", f"{data['n_repeats_genuinely_disagreeing']:,}"),
    ]
    table = ax.table(cellText=rows, colLabels=["Check", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title("Record-ID Duplicate Audit", pad=20)
    _add_caption(
        fig,
        "Every repeat is either a parse_error retried successfully, or a genuine same-paper "
        "duplicate identity already present in the source pool -- not reprocessing.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 3. Parse-error resolution
# ---------------------------------------------------------------------------

def parse_error_resolution_data(events: pd.DataFrame) -> dict:
    pe_ids = set(events.loc[events["classification"] == "parse_error", "record_id"])
    resolved = resolve_latest_per_record(events)
    still_unresolved = set(resolved.loc[resolved["classification"] == "parse_error", "record_id"])
    resolved_via_retry = pe_ids - still_unresolved
    return {
        "n_parse_error_events_logged": int((events["classification"] == "parse_error").sum()),
        "n_unique_records_that_ever_parse_errored": len(pe_ids),
        "n_resolved_via_retry": len(resolved_via_retry),
        "n_still_unresolved": len(still_unresolved),
        "still_unresolved_record_ids": sorted(still_unresolved),
    }


# ---------------------------------------------------------------------------
# 4. Curated (trusted, post-conflict-resolution) vs. landscape rate comparison
# ---------------------------------------------------------------------------

def curated_vs_landscape_comparison_data(
    canonical_df: pd.DataFrame, resolved_landscape: pd.DataFrame
) -> pd.DataFrame:
    trusted = canonical_df[
        canonical_df["label_confidence"].isin(_TRUSTED_LABEL_CONFIDENCE)
        & canonical_df["label"].isin(["positive", "negative"])
    ]
    c_pos, c_neg = int((trusted["label"] == "positive").sum()), int((trusted["label"] == "negative").sum())

    decided = resolved_landscape[resolved_landscape["classification"].isin(["positive", "negative"])]
    l_pos = int((decided["classification"] == "positive").sum())
    l_neg = int((decided["classification"] == "negative").sum())

    rows = [
        {"source": "Curated trusted set (post-resolution)", "n_positive": c_pos, "n_negative": c_neg,
         "n_total": c_pos + c_neg, "positive_rate": c_pos / (c_pos + c_neg) if (c_pos + c_neg) else None},
        {"source": "DeepSeek landscape (decided)", "n_positive": l_pos, "n_negative": l_neg,
         "n_total": l_pos + l_neg, "positive_rate": l_pos / (l_pos + l_neg) if (l_pos + l_neg) else None},
    ]
    return pd.DataFrame(rows)


def plot_curated_vs_landscape_comparison(
    canonical_df: pd.DataFrame, resolved_landscape: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    data = curated_vs_landscape_comparison_data(canonical_df, resolved_landscape)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = range(len(data))
    width = 0.35
    pos_bars = ax.bar([i - width / 2 for i in x], data["positive_rate"] * 100, width,
                       label="Positive %", color="#4C72B0")
    neg_bars = ax.bar([i + width / 2 for i in x], (1 - data["positive_rate"]) * 100, width,
                       label="Negative %", color="#C44E52")
    _bar_value_labels(ax, pos_bars, data["positive_rate"] * 100, fmt="{:.1f}%")
    _bar_value_labels(ax, neg_bars, (1 - data["positive_rate"]) * 100, fmt="{:.1f}%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["source"], fontsize=9)
    ax.set_ylabel("% of decided records")
    ax.set_title("Curated Trusted Set vs. DeepSeek Landscape -- Positive Rate")
    ax.legend()
    n_line = "  |  ".join(f"{row.source}: n={row.n_total:,}" for row in data.itertuples())
    _add_caption(
        fig,
        f"{n_line}. These are DISJOINT populations by design (the landscape run excluded every "
        "already-curated record before classifying) -- this is a rate/distribution comparison, "
        "NOT a per-record agreement or kappa score.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data
