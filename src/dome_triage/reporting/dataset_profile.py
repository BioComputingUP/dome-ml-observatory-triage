"""Dataset profiling & visualization suite (Step 18) -- profiles `canonical_dataset.csv` as a set
of charts: label distribution, journal diversity, provenance breakdown, BM25 score distribution,
year coverage, and how the curated sample compares to the full ~745k AI/ML-matched bulk pool it
was drawn from. None of this has previously existed as a single artifact.

Each chart is a pure `_*_data()` helper (testable without matplotlib or a display) wrapped by a
thin `plot_*()` rendering function, matching this project's existing convention (see
`pipeline/steps.py::_plot_clear_negative_score_distribution`). Titles stay short; anything that
needs real prose goes in a wrapped caption below the axes via `_add_caption` -- never crammed into
the title. The BM25 join reuses `curate/bulk_scores.py`'s existing lookup/annotate/threshold
helpers rather than reimplementing them -- including respecting that module's documented "never
call `Series.map()` on the 2.07M-entry lookup dict" performance rule (this module's frames are a
few thousand rows, well within the documented-safe regime for the per-row `.get()` loop
`annotate_bulk_scores` already uses).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

_LABEL_ORDER = ["positive", "negative", "skipped", "undeterminable"]
_LABEL_DISPLAY = {
    "positive": "Positive",
    "negative": "Negative",
    "skipped": "Skipped",
    "undeterminable": "Undeterminable",
}
_LABEL_COLORS = {
    "positive": "#4C72B0",
    "negative": "#C44E52",
    "skipped": "#8C8C8C",
    "undeterminable": "#8172B3",
}

_EPMC_NEGATIVE_SOURCE_NAMES = {
    "clear_negative_sampler",
    "clear_negative_sampler_strong",
    # Step 19d's review/non-methods-gated batch -- without this, these rows would silently fall
    # into "Other / Unscored" in the provenance breakdown chart instead of being separately
    # auditable as EPMC-sourced negatives.
    "clear_negative_sampler_strong_filtered_v2",
}

_PROVENANCE_CATEGORY_ORDER = [
    "Streamlit Curated",
    "Manual Curated (Pre-App)",
    "DOME Registry",
    "EPMC Negatives",
    "Other / Unscored",
]
_PROVENANCE_CATEGORY_COLORS = {
    "Streamlit Curated": "#4C72B0",
    "Manual Curated (Pre-App)": "#55A868",
    "DOME Registry": "#8172B3",
    "EPMC Negatives": "#C44E52",
    "Other / Unscored": "#8C8C8C",
}

_JOURNAL_BUCKETS: list[tuple[int, float, str]] = [
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 10, "6-10"),
    (11, 25, "11-25"),
    (26, 50, "26-50"),
    (51, 100, "51-100"),
    (101, float("inf"), "101+"),
]

_YEAR_MIN = 2000
_YEAR_MAX = 2026


def _mpl():
    """Local import with a forced non-interactive backend, matching the existing convention in
    `pipeline/steps.py` -- keeps matplotlib (and its backend probing) out of the import path for
    every other module that doesn't plot anything."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _add_caption(fig, text: str, width: int = 100) -> None:
    """Wraps `text` and places it below the axes, in the margin `tight_layout` reserves for it --
    the mechanism for keeping chart titles short: anything that needs real prose (methodology
    notes, caveats, exact figures) goes here instead of being crammed into `ax.set_title()`."""
    lines = textwrap.wrap(text, width=width) or [text]
    wrapped = "\n".join(lines)
    bottom = min(0.06 + 0.03 * len(lines), 0.42)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.5, bottom - 0.015, wrapped, ha="center", va="top", fontsize=8, color="#333333")


def _parse_source_names(sources_cell: object) -> list[str]:
    """Extracts every distinct `source_name` from one row's `sources` JSON-list column. Returns an
    empty list for missing/unparseable values rather than raising -- a handful of malformed rows
    should not abort the whole profiling run."""
    if not isinstance(sources_cell, str) or not sources_cell.strip():
        return []
    try:
        entries = json.loads(sources_cell)
    except json.JSONDecodeError:
        return []
    names = {e.get("source_name") for e in entries if isinstance(e, dict) and e.get("source_name")}
    return sorted(names)


def _to_year(series: pd.Series) -> pd.Series:
    """`year` is stored as a string, occasionally in the `"2012.0"` float-artifact form documented
    in Step 13's stratification explainer -- `pd.to_numeric` parses both forms; anything else
    becomes NaN and is naturally excluded by later `.dropna()`/groupby calls."""
    return pd.to_numeric(series, errors="coerce")


def _bar_value_labels(ax, bars, values, fmt="{:,}") -> None:
    """Exact numeric label centered above each bar -- applied consistently across every bar chart
    in this module, per the house rule that every chart shows real counts, not just shapes."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, fmt.format(int(value)),
            ha="center", va="bottom", fontsize=9,
        )


# ---------------------------------------------------------------------------
# 1. Label overview
# ---------------------------------------------------------------------------

def _label_overview_data(dataset: pd.DataFrame) -> pd.Series:
    counts = dataset["label"].value_counts()
    ordered = [label for label in _LABEL_ORDER if label in counts.index]
    extra = [label for label in counts.index if label not in _LABEL_ORDER]
    return counts.reindex(ordered + extra)


def plot_label_overview(dataset: pd.DataFrame, output_path: Path) -> pd.Series:
    counts = _label_overview_data(dataset)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [_LABEL_COLORS.get(label, "#8C8C8C") for label in counts.index]
    display_labels = [_LABEL_DISPLAY.get(label, label.title()) for label in counts.index]
    bars = ax.bar(display_labels, counts.values, color=colors, edgecolor="white")
    _bar_value_labels(ax, bars, counts.values)
    ax.set_xlabel("Label")
    ax.set_ylabel("Records")
    ax.set_title("Label Distribution")
    _add_caption(fig, f"Total records: {len(dataset):,}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# 2. Journal diversity
# ---------------------------------------------------------------------------

def _journal_bucket_label(record_count: int) -> str:
    for lo, hi, bucket_label in _JOURNAL_BUCKETS:
        if lo <= record_count <= hi:
            return bucket_label
    return _JOURNAL_BUCKETS[-1][2]


def _journal_diversity_data(dataset: pd.DataFrame) -> pd.Series:
    """Buckets journals by how many curated records they contribute, then counts how many
    distinct journals fall into each bucket -- directly answers "how concentrated is this dataset
    across journals" (e.g. "612 journals contribute exactly 1 record; 4 journals contribute
    100+"), which a top-20 list of the biggest journals cannot show at all -- it says nothing
    about the long tail, which is most of the diversity story here."""
    per_journal = dataset["journal"].value_counts()
    bucket_of = per_journal.apply(_journal_bucket_label)
    order = [bucket_label for _, _, bucket_label in _JOURNAL_BUCKETS]
    return bucket_of.value_counts().reindex(order, fill_value=0)


def plot_journal_diversity(dataset: pd.DataFrame, output_path: Path, top_n: int = 20) -> pd.Series:
    counts = _journal_diversity_data(dataset)
    per_journal = dataset["journal"].value_counts()
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
    _bar_value_labels(ax, bars, counts.values)
    ax.set_xlabel("Records Contributed By That Journal")
    ax.set_ylabel("Number Of Journals")
    ax.set_title("Journal Diversity")

    n_journals = len(per_journal)
    total_records = int(per_journal.sum())
    if n_journals:
        top_journal, top_journal_n = per_journal.index[0], int(per_journal.iloc[0])
        top_n_share = per_journal.head(top_n).sum() / total_records * 100 if total_records else 0.0
        caption = (
            f"{n_journals:,} distinct journals contribute {total_records:,} records. Most-represented: "
            f"{top_journal} ({top_journal_n:,} records). Top {top_n} journals account for "
            f"{top_n_share:.0f}% of all records."
        )
    else:
        caption = "No journal data available."
    _add_caption(fig, caption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# 3. Provenance breakdown
# ---------------------------------------------------------------------------

def _classify_provenance_category(
    label_confidence: object, source_names: set[str], is_streamlit_curated: bool
) -> str:
    """Priority order matters -- checked top to bottom, first match wins:

    1. Streamlit Curated: reviewed via the Curate app (`curation_events.csv`) -- the most recent,
       most authoritative human decision on record, even if the row originally entered via
       another pathway.
    2. EPMC Negatives: Step 14's clear-negative sampler. Checked before the confidence-tier rule
       below because these rows carry `label_confidence == "heuristic_candidate"` -- the same tier
       DOME-registry heuristic candidates use -- so without this earlier check they would be
       silently absorbed into the "DOME Registry" bucket (the bug an earlier version of this
       chart had).
    3. Manual Curated (Pre-App): `label_confidence == "human_curated"` but not in
       `curation_events.csv` -- the ~3,356-record cohort curated before this Curate app existed
       (`DOME_Top_Curate`, `copilot_1012`, etc.), same population Step 19 re-reviews.
    4. DOME Registry: `registry_confirmed`/`heuristic_candidate` confidence, everything else.
    5. Other / Unscored: whatever doesn't match any of the above (should be empty in practice)."""
    if is_streamlit_curated:
        return "Streamlit Curated"
    if source_names & _EPMC_NEGATIVE_SOURCE_NAMES:
        return "EPMC Negatives"
    if label_confidence == "human_curated":
        return "Manual Curated (Pre-App)"
    if label_confidence in ("registry_confirmed", "heuristic_candidate"):
        return "DOME Registry"
    return "Other / Unscored"


def _provenance_category_data(dataset: pd.DataFrame, streamlit_curated_ids) -> pd.Series:
    """Assigns every canonical record to exactly ONE provenance category, so the resulting bars
    sum to the full dataset row count -- unlike a per-`source_name` breakdown (this chart's
    earlier design), where the same record legitimately appears under >1 source after dedup and
    inflates naive per-source sums. See `_classify_provenance_category` for the exact rule."""
    streamlit_curated_ids = set(streamlit_curated_ids)
    categories = [
        _classify_provenance_category(
            label_confidence, set(_parse_source_names(sources_cell)), record_id in streamlit_curated_ids
        )
        for record_id, label_confidence, sources_cell in zip(
            dataset["record_id"], dataset["label_confidence"], dataset["sources"]
        )
    ]
    counts = pd.Series(categories).value_counts()
    ordered = [c for c in _PROVENANCE_CATEGORY_ORDER if c in counts.index]
    extra = [c for c in counts.index if c not in _PROVENANCE_CATEGORY_ORDER]
    return counts.reindex(ordered + extra)


def plot_provenance_class_breakdown(
    dataset: pd.DataFrame, output_path: Path, streamlit_curated_ids
) -> pd.Series:
    counts = _provenance_category_data(dataset, streamlit_curated_ids)
    total = int(counts.sum())
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [_PROVENANCE_CATEGORY_COLORS.get(c, "#8C8C8C") for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100 if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, f"{int(value):,} ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    if len(counts):
        ax.set_ylim(top=float(counts.values.max()) * 1.12)  # headroom so bar-top labels aren't cramped
    ax.set_xlabel("Provenance Category")
    ax.set_ylabel("Records")
    ax.set_title("Dataset Provenance Breakdown")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    caption = (
        "Each record counts in exactly one category (no double-counting). Streamlit Curated = "
        "reviewed via the Curate app. Manual Curated (Pre-App) = human-labeled before the app "
        "existed. DOME Registry = registry_confirmed/heuristic_candidate provenance, including "
        "source dome_registry_231_gold (231 records) and dome_registry_222_gold (222 records); "
        "cross-checked directly against canonical_dataset.csv: 222 of those 231 records carry "
        "both source tags (the two registry snapshots agree), the remaining 9 are unique to the "
        "231 snapshot, and the union is exactly 231 distinct records, confirmed NOT "
        "double-counted. EPMC Negatives = Step 14 clear-negative sampling."
    )
    _add_caption(fig, caption, width=112)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


# ---------------------------------------------------------------------------
# 4. BM25 score distribution by label
# ---------------------------------------------------------------------------

def _bm25_score_distribution_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """`dataset` must already carry a `bulk_match_score` column (via
    `curate/bulk_scores.py::annotate_bulk_scores`, called by the caller before this). Returns the
    positive/negative subset with a real (non-null) score -- records the bulk-match pool never
    scored (e.g. Step 14c's live-EPMC clear negatives) have no BM25 score and are excluded from
    this specific chart, not from the dataset as a whole."""
    scored = dataset[dataset["bulk_match_score"].notna()].copy()
    scored = scored[scored["label"].isin(["positive", "negative"])]
    scored["bulk_match_score"] = pd.to_numeric(scored["bulk_match_score"], errors="coerce")
    return scored[["label", "bulk_match_score"]].dropna()


def plot_bm25_score_distribution(
    dataset: pd.DataFrame, output_path: Path, threshold: float | None
) -> pd.DataFrame:
    import matplotlib.ticker as mticker

    scored = _bm25_score_distribution_data(dataset)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label in ("positive", "negative"):
        values = scored.loc[scored["label"] == label, "bulk_match_score"]
        if len(values):
            legend_label = f"{_LABEL_DISPLAY[label]} (min {values.min():.1f}, max {values.max():.1f})"
            ax.hist(values, bins=60, alpha=0.6, color=_LABEL_COLORS[label], label=legend_label)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="major", alpha=0.25)
    ax.grid(axis="x", which="minor", alpha=0.1)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(top=ymax * 1.15)
        ax.annotate(
            f"Youden Threshold: {threshold:.1f}", xy=(threshold, ymax * 1.03),
            ha="center", va="bottom", fontsize=9, fontweight="bold", annotation_clip=False,
        )

    ax.set_xlabel("BM25 Lexicon Score")
    ax.set_ylabel("Record Count")
    ax.set_title("BM25 Score Distribution By Label")
    ax.legend(loc="upper right")
    caption = (
        f"n={len(scored):,} records scored via the bulk-match pool join; records added through "
        "other pathways (e.g. EPMC clear negatives) were never scored and are excluded here. A "
        "negative scoring above the Youden threshold is a deliberate feature of the negative-"
        "sampling design, not an error: it was independently confirmed via a live EPMC query to "
        "not mention any AI/ML term despite the high lexicon score."
    )
    _add_caption(fig, caption, width=112)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return scored


# ---------------------------------------------------------------------------
# 5. Year distribution
# ---------------------------------------------------------------------------

def _year_distribution_data(
    dataset: pd.DataFrame, year_min: int = _YEAR_MIN, year_max: int = _YEAR_MAX
) -> pd.DataFrame:
    df = dataset.copy()
    df["year"] = _to_year(df["year"])
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    pivot = df.pivot_table(index="year", columns="label", values="record_id", aggfunc="count", fill_value=0)
    for col in _LABEL_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    extra_cols = [c for c in pivot.columns if c not in _LABEL_ORDER]
    pivot = pivot[_LABEL_ORDER + extra_cols]
    pivot = pivot.reindex(range(year_min, year_max + 1), fill_value=0)
    pivot.index.name = "year"
    return pivot


def plot_year_distribution(dataset: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    import matplotlib.ticker as mticker

    pivot = _year_distribution_data(dataset)
    n_pre_range = int((_to_year(dataset["year"]) < _YEAR_MIN).sum())
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, 7.5))
    bottom = [0] * len(pivot)
    for label_key in pivot.columns:
        color = _LABEL_COLORS.get(label_key, "#8C8C8C")
        display = _LABEL_DISPLAY.get(label_key, str(label_key).title())
        ax.bar(pivot.index, pivot[label_key], bottom=bottom, label=display, color=color)
        bottom = [b + v for b, v in zip(bottom, pivot[label_key])]
    max_total = max(bottom) if bottom else 0
    ax.set_ylim(top=max_total * 1.2 if max_total else 1)  # headroom above the tallest stacked bar
    ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.set_xticks(list(pivot.index))
    ax.set_xticklabels([str(y) for y in pivot.index], rotation=90, fontsize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Records")
    ax.set_title("Records Per Year By Label")
    ax.legend()
    caption = (
        f"Years {_YEAR_MIN}-{_YEAR_MAX} shown. {n_pre_range} pre-{_YEAR_MIN} record(s) exist "
        f"elsewhere in the dataset and are excluded from this chart."
    )
    _add_caption(fig, caption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return pivot


# ---------------------------------------------------------------------------
# 6. Year coverage vs. the full bulk pool
# ---------------------------------------------------------------------------

def _year_coverage_vs_bulk_pool_data(
    dataset: pd.DataFrame, bulk_pool_years: pd.Series, year_min: int = _YEAR_MIN, year_max: int = _YEAR_MAX
) -> pd.DataFrame:
    """`bulk_pool_years` is the raw `year` column of the ~745k-row bulk-candidates-scored pool
    (read by the caller with `usecols=["year"]` only -- this function never sees the other 26
    columns of that file). Returns one row per year in `[year_min, year_max]`, with `curated` and
    `pool` counts and the `coverage_pct` the curated sample represents of that year's pool."""
    curated_years = _to_year(dataset["year"]).dropna().astype(int)
    pool_years = _to_year(bulk_pool_years).dropna().astype(int)

    curated_counts = curated_years.value_counts()
    pool_counts = pool_years.value_counts()

    all_years = range(year_min, year_max + 1)
    out = pd.DataFrame(index=all_years)
    out["curated"] = curated_counts.reindex(all_years, fill_value=0)
    out["pool"] = pool_counts.reindex(all_years, fill_value=0)
    out["coverage_pct"] = (out["curated"] / out["pool"].replace(0, pd.NA)) * 100
    out.index.name = "year"
    return out


def _plot_year_coverage_vs_bulk_pool(
    dataset: pd.DataFrame, bulk_pool_years: pd.Series, output_path: Path, log_scale: bool
) -> pd.DataFrame:
    coverage = _year_coverage_vs_bulk_pool_data(dataset, bulk_pool_years)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(coverage.index, coverage["pool"], color="#B0B0B0", label="Full AI/ML Bulk Pool (~745k)")
    ax.bar(coverage.index, coverage["curated"], color="#4C72B0", label="Curated Dataset")
    ax.set_xticks(list(coverage.index))
    ax.set_xticklabels([str(y) for y in coverage.index], rotation=90, fontsize=7)
    ax.set_xlabel("Year")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Records (Log Scale)")
        ax.set_title("Curated Coverage Vs. Full AI/ML Bulk Pool By Year (Log Scale)")
        caption = (
            f"Years {_YEAR_MIN}-{_YEAR_MAX}, log scale. The curated sample is a small fraction of "
            "most years' true AI/ML-matched population. That is expected: this dataset is a "
            "stratified sample plus targeted negatives, not an attempt to curate the full pool."
        )
    else:
        ax.set_ylabel("Records")
        ax.set_title("Curated Coverage Vs. Full AI/ML Bulk Pool By Year (Linear Scale)")
        caption = (
            f"Years {_YEAR_MIN}-{_YEAR_MAX}, linear scale. True to scale, so the curated (blue) "
            "bars are visually dwarfed by the full bulk pool (grey) in every year. See the "
            "log-scale version of this chart to compare the curated sample's own shape across "
            "years without that scale difference hiding it."
        )
    ax.legend()
    _add_caption(fig, caption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return coverage


def plot_year_coverage_vs_bulk_pool(
    dataset: pd.DataFrame, bulk_pool_years: pd.Series, output_path: Path
) -> pd.DataFrame:
    return _plot_year_coverage_vs_bulk_pool(dataset, bulk_pool_years, output_path, log_scale=True)


def plot_year_coverage_vs_bulk_pool_linear(
    dataset: pd.DataFrame, bulk_pool_years: pd.Series, output_path: Path
) -> pd.DataFrame:
    return _plot_year_coverage_vs_bulk_pool(dataset, bulk_pool_years, output_path, log_scale=False)


# ---------------------------------------------------------------------------
# 7. BM25 Youden-threshold performance vs. human curation (Q1-Q4 stratified queue)
# ---------------------------------------------------------------------------

def _bm25_youden_performance_data(joined: pd.DataFrame) -> pd.DataFrame:
    """`joined` is one row per Step-13 quartile-stratified-queue record with `label` (the current
    human-curated ground truth, already restricted by the caller to positive/negative), `quartile`
    (1-4, 1 = lowest BM25 score, 4 = highest -- the same Q1-Q4 bands the main Curate page's score
    band filter uses), `match_classification__bm25` (Step 12's Youden-threshold call, already
    restricted to positive/negative), and `match_score__bm25` (the raw BM25 score each quartile
    band was actually computed from). Returns one row per quartile plus an "All" row, with
    correct/incorrect counts, accuracy_pct -- how often the Youden threshold's call actually agreed
    with the human curator, measured directly against the real curation outcomes -- and each
    quartile's real BM25 score range/average (`score_min`/`score_max`/`score_avg`), so a reader
    can see exactly what score band "Q3", say, actually covers, not just its label."""
    df = joined.copy()
    df["correct"] = df["label"] == df["match_classification__bm25"]
    df["match_score__bm25"] = pd.to_numeric(df["match_score__bm25"], errors="coerce")

    rows = []
    for quartile in sorted(df["quartile"].dropna().unique()):
        sub = df[df["quartile"] == quartile]
        rows.append(
            {
                "quartile": f"Q{int(quartile)}",
                "correct": int(sub["correct"].sum()),
                "incorrect": int((~sub["correct"]).sum()),
                "total": int(len(sub)),
                "accuracy_pct": float(sub["correct"].mean() * 100) if len(sub) else 0.0,
                "score_min": float(sub["match_score__bm25"].min()) if len(sub) else float("nan"),
                "score_max": float(sub["match_score__bm25"].max()) if len(sub) else float("nan"),
                "score_avg": float(sub["match_score__bm25"].mean()) if len(sub) else float("nan"),
            }
        )
    rows.append(
        {
            "quartile": "All",
            "correct": int(df["correct"].sum()),
            "incorrect": int((~df["correct"]).sum()),
            "total": int(len(df)),
            "accuracy_pct": float(df["correct"].mean() * 100) if len(df) else 0.0,
            "score_min": float(df["match_score__bm25"].min()) if len(df) else float("nan"),
            "score_max": float(df["match_score__bm25"].max()) if len(df) else float("nan"),
            "score_avg": float(df["match_score__bm25"].mean()) if len(df) else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def _bm25_youden_confusion_matrix_data(joined: pd.DataFrame) -> pd.DataFrame:
    """2x2 confusion matrix: index = human-curated label, columns = BM25 Youden-threshold call.
    Both axes ordered [positive, negative] (when both are present) so the diagonal
    (top-left/bottom-right) is always the "correct" cells, regardless of crosstab's default
    alphabetical ordering."""
    matrix = pd.crosstab(joined["label"], joined["match_classification__bm25"])
    order = [label for label in ("positive", "negative") if label in matrix.index]
    col_order = [label for label in ("positive", "negative") if label in matrix.columns]
    return matrix.reindex(index=order, columns=col_order, fill_value=0)


def plot_bm25_youden_performance(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    performance = _bm25_youden_performance_data(joined)
    confusion = _bm25_youden_confusion_matrix_data(joined)
    all_row = performance[performance["quartile"] == "All"].iloc[0]
    total, overall_acc = int(all_row["total"]), float(all_row["accuracy_pct"])

    def _cell(row_label: str, col_label: str) -> int:
        if row_label in confusion.index and col_label in confusion.columns:
            return int(confusion.loc[row_label, col_label])
        return 0

    tp, fn = _cell("positive", "positive"), _cell("positive", "negative")
    fp, tn = _cell("negative", "positive"), _cell("negative", "negative")
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) else 0.0

    plt = _mpl()
    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(15, 6.5))

    cm_values = confusion.values
    ax_cm.imshow(cm_values, cmap="Blues")
    ax_cm.set_xticks(range(len(confusion.columns)))
    ax_cm.set_xticklabels([c.title() for c in confusion.columns])
    ax_cm.set_yticks(range(len(confusion.index)))
    ax_cm.set_yticklabels([i.title() for i in confusion.index])
    ax_cm.set_xlabel("BM25 Youden-Threshold Call")
    ax_cm.set_ylabel("Human-Curated Label")
    ax_cm.set_title("Confusion Matrix")
    cm_max = cm_values.max() if cm_values.size else 0
    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            value = cm_values[i, j]
            pct = value / total * 100 if total else 0.0
            color = "white" if cm_max and value > cm_max * 0.5 else "black"
            ax_cm.text(
                j, i, f"{value:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11
            )

    # Right panel: accuracy by quartile, with each bar's own Youden-score range/average labeled
    # directly on the bar -- not just the Qn name -- so "Q3", say, is never an opaque label; the
    # real BM25 score band it covers is always visible right there.
    quartile_rows = performance[performance["quartile"] != "All"]
    colors = ["#55A868" if acc >= overall_acc else "#C44E52" for acc in quartile_rows["accuracy_pct"]]
    bars = ax_bar.bar(quartile_rows["quartile"], quartile_rows["accuracy_pct"], color=colors)
    for bar, row in zip(bars, quartile_rows.itertuples()):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, row.accuracy_pct + 2,
            f"{row.accuracy_pct:.1f}%  (n={row.total})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, row.accuracy_pct / 2,
            f"BM25 {row.score_min:.1f}-{row.score_max:.1f}\navg {row.score_avg:.1f}",
            ha="center", va="center", fontsize=9, color="white",
        )
    ax_bar.axhline(overall_acc, color="black", linestyle="--", linewidth=1, label=f"Overall: {overall_acc:.1f}%")
    ax_bar.set_ylim(top=115)
    ax_bar.set_xlabel("BM25 Score Quartile (Q1 = Lowest, Q4 = Highest)")
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_title("Accuracy By Quartile\n(bar label: accuracy/n; in-bar: Youden score range and average)")
    ax_bar.legend(loc="lower right")

    fig.suptitle("BM25 Youden-Threshold Performance Vs. Human Curation")
    provenance_note = (
        "Provenance: every record with a decision in curation_events.csv (curated through the "
        "Streamlit Curate app -- any queue source, not only the Step 13 stratified sample) that "
        "currently has a definitive positive/negative label and a BM25 score. Quartiles are "
        "computed fresh over exactly that population (the same build_strata/pd.qcut method the "
        "live Curate app itself uses for its own Q1-Q4 bands), not read from a stale one-time "
        "snapshot. "
    )
    caption = (
        provenance_note
        + f"n={total:,} records. Overall accuracy {overall_acc:.1f}%. Precision {precision:.1f}% "
        f"(of records BM25 called positive, how many the human curator also called positive). "
        f"Recall {recall:.1f}% (of records the human curator called positive, how many BM25's "
        f"threshold also caught). Specificity {specificity:.1f}% (of records the human curator "
        f"called negative, how many BM25's threshold also called negative)."
    )
    _add_caption(fig, caption, width=130)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return performance


# ---------------------------------------------------------------------------
# 9. Original-cohort second review (Step 19b, additive columns) -- new folder, not
#    dataset_profile/, since this reports on a distinct sub-cohort/sub-question, not the whole
#    dataset. See `merge_original_cohort_second_review` in curate/state.py for how the
#    `original_cohort_review_*` columns these charts read actually got onto canonical_dataset.csv
#    -- additively, never overwriting the pre-existing `label`/`label_confidence`.
# ---------------------------------------------------------------------------

_SECOND_REVIEW_LABEL_ORDER = ["negative", "positive", "undeterminable", "skipped"]


def _second_review_decision_breakdown_data(dataset: pd.DataFrame) -> pd.Series:
    """`dataset` must already carry `original_cohort_review_label` (written by
    `merge_original_cohort_second_review`). Counts the fresh, independent second-review decision
    itself -- e.g. how many of the original cohort's negatives were reconfirmed negative vs.
    flipped to positive on this second look -- restricted to rows this review pass actually
    covered (NaN everywhere else, since it's an additive merge over a sub-cohort, not the whole
    dataset)."""
    reviewed = dataset["original_cohort_review_label"].dropna()
    counts = reviewed.value_counts()
    ordered = [label for label in _SECOND_REVIEW_LABEL_ORDER if label in counts.index]
    extra = [label for label in counts.index if label not in _SECOND_REVIEW_LABEL_ORDER]
    return counts.reindex(ordered + extra)


def plot_second_review_decision_breakdown(dataset: pd.DataFrame, output_path: Path) -> pd.Series:
    counts = _second_review_decision_breakdown_data(dataset)
    total = int(counts.sum())
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [_LABEL_COLORS.get(label, "#8C8C8C") for label in counts.index]
    display_labels = [_LABEL_DISPLAY.get(label, label.title()) for label in counts.index]
    bars = ax.bar(display_labels, counts.values, color=colors, edgecolor="white")
    ax.set_ylim(top=(counts.max() if len(counts) else 0) * 1.15)
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100 if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("Second-Review Decision")
    ax.set_ylabel("Records")
    ax.set_title("Original-Cohort Negatives: Second-Review Outcome")

    reviewed_mask = dataset["original_cohort_review_label"].notna()
    prior_labels = dataset.loc[reviewed_mask, "original_cohort_review_prior_label"]
    prior_all_negative = bool((prior_labels == "negative").all()) if len(prior_labels) else False
    caption = (
        f"n={total:,} original-cohort records independently re-reviewed (Step 19, "
        f"\"Original Cohort Review\" page). Every one of these rows carried a prior trusted "
        f"`label` of {'negative (100%)' if prior_all_negative else 'a mix -- see original_cohort_review_prior_label'} "
        f"before this second look. This chart counts the fresh decision only -- see "
        f"original_cohort_review_agrees_with_prior in canonical_dataset.csv for the per-record "
        f"agree/disagree flag. The original label/label_confidence columns are untouched by this "
        f"merge (additive columns only -- see AGENTS.md/STEPS_Progress.md Step 19b)."
    )
    _add_caption(fig, caption, width=110)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def _second_review_bm25_confusion_data(joined: pd.DataFrame) -> pd.DataFrame:
    """`joined`: one row per second-reviewed record already restricted to a positive/negative
    `original_cohort_review_label` (undeterminable/skipped aren't real ground truth) and a
    `match_classification__bm25` call. Same [positive, negative] axis-ordering convention as
    `_bm25_youden_confusion_matrix_data`."""
    matrix = pd.crosstab(joined["original_cohort_review_label"], joined["match_classification__bm25"])
    order = [label for label in ("positive", "negative") if label in matrix.index]
    col_order = [label for label in ("positive", "negative") if label in matrix.columns]
    return matrix.reindex(index=order, columns=col_order, fill_value=0)


def plot_second_review_bm25_confusion(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    confusion = _second_review_bm25_confusion_data(joined)
    total = int(confusion.values.sum())

    def _cell(row_label: str, col_label: str) -> int:
        if row_label in confusion.index and col_label in confusion.columns:
            return int(confusion.loc[row_label, col_label])
        return 0

    tp, fn = _cell("positive", "positive"), _cell("positive", "negative")
    fp, tn = _cell("negative", "positive"), _cell("negative", "negative")
    accuracy = (tp + tn) / total * 100 if total else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) else 0.0
    # Scored instead against the STALE pre-review assumption that every one of these records is
    # still "negative": only BM25's own "negative" calls would count as correct, regardless of
    # what the (not-yet-known) real label was -- (fn + tn) / total, not specificity (a rate, not a
    # share of the whole population).
    stale_accuracy = (fn + tn) / total * 100 if total else 0.0

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cm_values = confusion.values
    ax.imshow(cm_values, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_xticklabels([c.title() for c in confusion.columns])
    ax.set_yticks(range(len(confusion.index)))
    ax.set_yticklabels([i.title() for i in confusion.index])
    ax.set_xlabel("BM25 Youden-Threshold Call")
    ax.set_ylabel("Second-Review Label (Corrected Ground Truth)")
    ax.set_title("BM25 Vs. Second-Reviewed Original-Cohort Negatives")
    cm_max = cm_values.max() if cm_values.size else 0
    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            value = cm_values[i, j]
            pct = value / total * 100 if total else 0.0
            color = "white" if cm_max and value > cm_max * 0.5 else "black"
            ax.text(
                j, i, f"{value:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11
            )

    caption = (
        f"n={total:,} original-cohort records with both a definitive second-review label "
        f"(undeterminable/skipped excluded) and a BM25 score. Accuracy against the corrected "
        f"(second-review) ground truth: {accuracy:.1f}%. Precision {precision:.1f}%, recall "
        f"{recall:.1f}%, specificity {specificity:.1f}%. For context: scored instead against the "
        f"stale pre-review assumption that every one of these records is still \"negative\" (only "
        f"counting BM25's own \"negative\" calls as correct), BM25 would read as "
        f"{stale_accuracy:.1f}% accurate -- the {tp:,} records BM25's threshold already called "
        f"positive, which this second review now also calls positive, are exactly where the two "
        f"figures diverge."
    )
    _add_caption(fig, caption, width=120)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return confusion


def _second_review_bm25_score_distribution_data(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    df["match_score__bm25"] = pd.to_numeric(df["match_score__bm25"], errors="coerce")
    return df[["original_cohort_review_label", "match_score__bm25"]].dropna()


def plot_second_review_bm25_score_distribution(
    joined: pd.DataFrame, output_path: Path, threshold: float | None
) -> pd.DataFrame:
    data = _second_review_bm25_score_distribution_data(joined)
    negative_scores = data.loc[data["original_cohort_review_label"] == "negative", "match_score__bm25"]
    positive_scores = data.loc[data["original_cohort_review_label"] == "positive", "match_score__bm25"]

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        negative_scores, bins=40, alpha=0.6, color="#C44E52",
        label=f"Reconfirmed negative (n={len(negative_scores):,})",
    )
    ax.hist(
        positive_scores, bins=40, alpha=0.6, color="#4C72B0",
        label=f"Flipped to positive (n={len(positive_scores):,})",
    )
    if threshold is not None:
        ax.axvline(
            threshold, color="black", linestyle="--", linewidth=1.5,
            label=f"Youden threshold ({threshold:.1f})",
        )
    ax.set_xlabel("BM25 Match Score")
    ax.set_ylabel("Records")
    ax.set_title("BM25 Score: Reconfirmed-Negative Vs. Flipped-To-Positive")
    ax.legend()
    caption = (
        f"Original-cohort negatives with a BM25 score, split by this second review's outcome. If "
        f"the flipped-to-positive group's scores skew higher / cluster more above the Youden "
        f"threshold than the reconfirmed-negative group's, that's independent evidence BM25 was "
        f"already picking up real signal in these {len(positive_scores):,} records that the "
        f"original manual curation missed, not just noise."
    )
    _add_caption(fig, caption, width=115)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 10. Step 19d -- likely-review/non-methods flag vs. label, and what's actually triggering it
# ---------------------------------------------------------------------------


_NOT_FLAGGED_COL = "not_flagged"
_FLAGGED_COL = "flagged"


def _label_vs_review_flag_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of `label` x `likely_review_or_non_methods` -- how many of each label are
    flagged vs. not, computed dataset-wide (`curate/state.py::flag_likely_reviews`'s output, not
    scoped to any one batch). Empty (all-zero) if the flag column doesn't exist yet.

    Columns are the strings `_NOT_FLAGGED_COL`/`_FLAGGED_COL`, deliberately never the bare
    booleans `False`/`True` -- a real, confirmed pandas gotcha caught while building this: on a
    DataFrame whose column *labels* are the literal booleans `False`/`True`,
    `df[[False, True]]` does not reliably mean "select the columns labeled False and True" --
    verified live that when the row count happens to equal the list length (2, here), pandas
    instead applies `[False, True]` as a *positional boolean row mask*, silently dropping the
    first row. String column labels sidestep the ambiguity entirely."""
    if "likely_review_or_non_methods" not in dataset.columns:
        return pd.DataFrame(columns=[_NOT_FLAGGED_COL, _FLAGGED_COL])

    df = dataset.dropna(subset=["likely_review_or_non_methods"]).copy()
    df["flagged"] = df["likely_review_or_non_methods"].astype(str) == "True"
    counts = df.groupby(["label", "flagged"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={False: _NOT_FLAGGED_COL, True: _FLAGGED_COL})
    for col in (_NOT_FLAGGED_COL, _FLAGGED_COL):
        if col not in counts.columns:
            counts[col] = 0

    ordered = [label for label in _LABEL_ORDER if label in counts.index]
    ordered += [label for label in counts.index if label not in _LABEL_ORDER]
    return counts.reindex(ordered)[[_NOT_FLAGGED_COL, _FLAGGED_COL]]


def plot_label_vs_review_flag_breakdown(dataset: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    data = _label_vs_review_flag_data(dataset)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 6))

    if data.empty:
        ax.text(0.5, 0.5, "No likely_review_or_non_methods data available", ha="center", va="center")
        ax.axis("off")
    else:
        display_labels = [_LABEL_DISPLAY.get(label, str(label).title()) for label in data.index]
        positions = list(range(len(data.index)))
        width = 0.35
        not_flagged, flagged = data[_NOT_FLAGGED_COL].values, data[_FLAGGED_COL].values
        bars_clean = ax.bar(
            [p - width / 2 for p in positions], not_flagged, width, label="Not flagged", color="#4C72B0"
        )
        bars_flagged = ax.bar(
            [p + width / 2 for p in positions], flagged, width,
            label="Likely review / non-methods", color="#C44E52",
        )
        _bar_value_labels(ax, bars_clean, not_flagged)
        _bar_value_labels(ax, bars_flagged, flagged)
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels)
        ax.legend()

    ax.set_xlabel("Label")
    ax.set_ylabel("Records")
    ax.set_title("Label Vs. Likely-Review/Non-Methods Flag")

    total = int(data.values.sum()) if not data.empty else 0
    caption_parts = [
        f"n={total:,} records with a likely_review_or_non_methods flag computed (Step 19d's "
        "NLTK-based non-methods detector: title/abstract text-lexicon match OR a non-methods "
        "EPMC pub_types tag -- either signal alone is sufficient)."
    ]
    for label in ("positive", "negative"):
        if label in data.index:
            label_total = int(data.loc[label].sum())
            label_flagged = int(data.loc[label, _FLAGGED_COL])
            pct = label_flagged / label_total * 100 if label_total else 0.0
            caption_parts.append(
                f"{_LABEL_DISPLAY.get(label, label.title())}: {label_flagged:,}/{label_total:,} "
                f"({pct:.1f}%) flagged."
            )
    _add_caption(fig, " ".join(caption_parts), width=110)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


def _review_flag_term_frequency_data(dataset: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Tallies which text terms / pub_types tags actually fired across flagged rows, parsed from
    `likely_review_or_non_methods_detail` -- so the flag isn't a black box. Pub-type hits are
    suffixed `" (pub_type)"` to keep them visually distinct from lemmatized text-term hits in the
    same ranking (e.g. "review" the text term vs. "Review" the pub_types tag)."""
    if "likely_review_or_non_methods_detail" not in dataset.columns:
        return pd.Series(dtype=int)

    counts: dict[str, int] = {}
    for cell in dataset["likely_review_or_non_methods_detail"].dropna():
        if isinstance(cell, str):
            try:
                detail = json.loads(cell)
            except json.JSONDecodeError:
                continue
        else:
            detail = cell
        if not isinstance(detail, dict):
            continue
        for hit in detail.get("text_hits") or []:
            counts[hit] = counts.get(hit, 0) + 1
        for hit in detail.get("pub_type_hits") or []:
            key = f"{hit} (pub_type)"
            counts[key] = counts.get(key, 0) + 1

    return pd.Series(counts, dtype=int).sort_values(ascending=False).head(top_n)


def plot_review_flag_term_frequency(dataset: pd.DataFrame, output_path: Path, top_n: int = 15) -> pd.Series:
    data = _review_flag_term_frequency_data(dataset, top_n)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(data) + 1)))

    if data.empty:
        ax.text(0.5, 0.5, "No flagged records / no detail data available", ha="center", va="center")
        ax.axis("off")
    else:
        # Reversed so the highest-frequency term renders at the top of the horizontal bar chart.
        labels, values = data.index[::-1], data.values[::-1]
        bars = ax.barh(labels, values, color="#55A868")
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,}", va="center", fontsize=9)
        ax.set_xlabel("Flagged Records")
        ax.set_ylabel("Matched Term / Pub Type")

    ax.set_title(f"Top {top_n} Signals Behind The Likely-Review Flag")
    caption = (
        "Which lemmatized text terms (e.g. \"meta analysis\" covers both \"meta-analysis\" and "
        "\"meta analysis\" in the source text) and EPMC pub_types tags actually triggered the flag "
        "most often -- a direct check that the gate is catching genuine review/non-methods "
        "content, not an opaque black box."
    )
    _add_caption(fig, caption, width=110)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data
