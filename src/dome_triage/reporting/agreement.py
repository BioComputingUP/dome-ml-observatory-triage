"""Step 20: human-vs-DeepSeek agreement metrics and charts -- Cohen's kappa
(`sklearn.metrics.cohen_kappa_score`, first use in this codebase; `scikit-learn>=1.4` is already a
dependency, nothing new to install), confusion matrices, and the undeterminable-subset fallback
comparison (see STEPS_Progress.md Step 20, "Undeterminable rate and RAG fallback").

Same pure `_*_data()` helper + thin `plot_*()` renderer convention as `dataset_profile.py`; reuses
that module's `_mpl`/`_add_caption`/`_bar_value_labels` directly (intra-package import, no
duplication -- see that module for the shared reasoning) rather than a second copy.

`joined` throughout this module means: one row per sampled record with `label` (the human's real,
trusted positive/negative decision) and `classification` (DeepSeek's answer for one tier/mode --
positive/negative/undeterminable/parse_error). Rows where `classification == PARSE_ERROR` are
excluded from every kappa/accuracy computation, never coerced into a guessed label, and reported
as their own explicit count.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from dome_triage.llm_classify.response_parser import PARSE_ERROR
from dome_triage.reporting.dataset_profile import _add_caption, _bar_value_labels, _mpl

_LLM_LABELS = ("positive", "negative", "undeterminable")
_HUMAN_LABELS = ("positive", "negative")


# ---------------------------------------------------------------------------
# 1. Per-tier agreement summary (kappa, accuracy, undetermined/parse-error rates)
# ---------------------------------------------------------------------------


def compute_agreement(joined: pd.DataFrame, tier: str) -> dict:
    n_total = int(len(joined))
    n_parse_error = int((joined["classification"] == PARSE_ERROR).sum())
    scored = joined[joined["classification"] != PARSE_ERROR]

    kappa = (
        float(cohen_kappa_score(scored["label"], scored["classification"], labels=list(_LLM_LABELS)))
        if len(scored)
        else float("nan")
    )
    n_undetermined = int((scored["classification"] == "undeterminable").sum())
    decided = scored[scored["classification"].isin(_HUMAN_LABELS)]
    accuracy_among_decided = (
        float((decided["label"] == decided["classification"]).mean()) if len(decided) else float("nan")
    )

    return {
        "tier": tier,
        "n_total": n_total,
        "n_scored": int(len(scored)),
        "n_parse_error": n_parse_error,
        "n_undetermined": n_undetermined,
        "undetermined_rate": (n_undetermined / len(scored)) if len(scored) else float("nan"),
        "n_decided": int(len(decided)),
        "accuracy_among_decided": accuracy_among_decided,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# 2. Confusion matrix per tier -- 2 (human) x 3 (LLM) rows/cols, never assumed square
# ---------------------------------------------------------------------------


def _confusion_matrix_data(joined: pd.DataFrame) -> pd.DataFrame:
    scored = joined[joined["classification"] != PARSE_ERROR]
    matrix = pd.crosstab(scored["label"], scored["classification"])
    row_order = [label for label in _HUMAN_LABELS if label in matrix.index]
    col_order = [label for label in _LLM_LABELS if label in matrix.columns]
    return matrix.reindex(index=row_order, columns=col_order, fill_value=0)


def plot_confusion_matrix_per_tier(joined: pd.DataFrame, tier: str, output_path) -> pd.DataFrame:
    confusion = _confusion_matrix_data(joined)
    agreement = compute_agreement(joined, tier)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cm_values = confusion.values
    ax.imshow(cm_values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_xticklabels([c.title() for c in confusion.columns])
    ax.set_yticks(range(len(confusion.index)))
    ax.set_yticklabels([i.title() for i in confusion.index])
    ax.set_xlabel(f"DeepSeek ({tier}) Classification")
    ax.set_ylabel("Human-Curated Label")
    ax.set_title(f"Human Vs. DeepSeek ({tier}) Confusion Matrix")

    total = int(cm_values.sum())
    cm_max = cm_values.max() if cm_values.size else 0
    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            value = cm_values[i, j]
            pct = value / total * 100 if total else 0.0
            color = "white" if cm_max and value > cm_max * 0.5 else "black"
            ax.text(j, i, f"{value:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11)

    caption = (
        f"n={agreement['n_scored']:,} scored ({agreement['n_parse_error']:,} parse_error rows "
        f"excluded entirely, never coerced into a guess). Cohen's kappa={agreement['kappa']:.3f}. "
        f"Among the {agreement['n_decided']:,} records DeepSeek answered positive/negative (not "
        f"undeterminable), accuracy vs. the human label was {agreement['accuracy_among_decided'] * 100:.1f}%. "
        f"Undetermined rate: {agreement['undetermined_rate'] * 100:.1f}% "
        f"({agreement['n_undetermined']:,} records)."
    )
    _add_caption(fig, caption, width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return confusion


# ---------------------------------------------------------------------------
# 3. Kappa/accuracy summary across whichever tiers were actually run
# ---------------------------------------------------------------------------


def plot_kappa_accuracy_summary(agreements: dict, output_path) -> pd.DataFrame:
    """`agreements`: {tier_name: compute_agreement(...) dict}, one entry per tier actually run
    (may be just one tier, e.g. flash-only during an early check)."""
    rows = [
        {"tier": tier, "kappa": a["kappa"], "accuracy_among_decided": a["accuracy_among_decided"]}
        for tier, a in agreements.items()
    ]
    data = pd.DataFrame(rows)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(len(data))
    width = 0.35
    bars_kappa = ax.bar(
        [p - width / 2 for p in positions], data["kappa"], width, label="Cohen's Kappa", color="#4C72B0"
    )
    bars_acc = ax.bar(
        [p + width / 2 for p in positions], data["accuracy_among_decided"], width,
        label="Accuracy (among decided)", color="#55A868",
    )
    for bar, value in zip(bars_kappa, data["kappa"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars_acc, data["accuracy_among_decided"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(data["tier"]))
    ax.set_ylim(top=1.15)
    ax.set_ylabel("Score")
    ax.set_title("Human Vs. DeepSeek Agreement By Tier")
    ax.legend()
    _add_caption(
        fig,
        "Kappa scored over every record with a parseable answer (undeterminable included as its "
        "own class); accuracy scored only over records DeepSeek actually committed to "
        "positive/negative -- these two numbers answer different questions and are not directly "
        "comparable to each other.",
        width=105,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 4. Flash vs. Pro cross-tier comparison
# ---------------------------------------------------------------------------


def _flash_vs_pro_comparison_data(joined_flash: pd.DataFrame, joined_pro: pd.DataFrame) -> pd.DataFrame:
    """Only records BOTH tiers classified (inner join on record_id) with a parseable answer from
    both. Buckets each record into exactly one of: both tiers agree with each other and match the
    human label; both agree with each other but neither matches the human label; the tiers
    disagree with each other and flash matches the human label; disagree and pro matches; disagree
    and neither matches."""
    left = joined_flash[joined_flash["classification"] != PARSE_ERROR][["record_id", "label", "classification"]]
    right = joined_pro[joined_pro["classification"] != PARSE_ERROR][["record_id", "classification"]]
    merged = left.merge(right, on="record_id", suffixes=("_flash", "_pro"))

    def _bucket(row) -> str:
        tiers_agree = row["classification_flash"] == row["classification_pro"]
        flash_correct = row["classification_flash"] == row["label"]
        pro_correct = row["classification_pro"] == row["label"]
        if tiers_agree:
            return "Both Agree, Match Human" if flash_correct else "Both Agree, Neither Matches Human"
        if flash_correct:
            return "Disagree, Flash Matches Human"
        if pro_correct:
            return "Disagree, Pro Matches Human"
        return "Disagree, Neither Matches Human"

    merged["bucket"] = merged.apply(_bucket, axis=1)
    order = [
        "Both Agree, Match Human",
        "Both Agree, Neither Matches Human",
        "Disagree, Flash Matches Human",
        "Disagree, Pro Matches Human",
        "Disagree, Neither Matches Human",
    ]
    counts = merged["bucket"].value_counts().reindex(order, fill_value=0)
    return counts.to_frame(name="count")


def plot_flash_vs_pro_comparison(joined_flash: pd.DataFrame, joined_pro: pd.DataFrame, output_path) -> pd.DataFrame:
    data = _flash_vs_pro_comparison_data(joined_flash, joined_pro)
    total = int(data["count"].sum())

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#55A868", "#C44E52", "#4C72B0", "#8172B3", "#8C8C8C"]
    bars = ax.barh(data.index[::-1], data["count"].values[::-1], color=colors[::-1])
    for bar, value in zip(bars, data["count"].values[::-1]):
        pct = value / total * 100 if total else 0.0
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {int(value):,} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Records")
    ax.set_title("Flash Vs. Pro: Cross-Tier Agreement On The Same Sample")
    _add_caption(
        fig,
        f"n={total:,} records both tiers classified with a parseable answer. Both-tier agreement "
        "with each other is a ceiling on how far either alone can be trusted -- it does not by "
        "itself confirm either is right, only that they're consistent.",
        width=105,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 5. Disagreement breakdown (off-diagonal confusion-matrix cells) for one tier
# ---------------------------------------------------------------------------


def _disagreement_breakdown_data(joined: pd.DataFrame) -> pd.DataFrame:
    confusion = _confusion_matrix_data(joined)
    rows = []
    for human_label in confusion.index:
        for llm_label in confusion.columns:
            if human_label == llm_label:
                continue
            count = int(confusion.loc[human_label, llm_label])
            if count:
                rows.append({"human_label": human_label, "llm_classification": llm_label, "count": count})
    return pd.DataFrame(rows, columns=["human_label", "llm_classification", "count"]).sort_values(
        "count", ascending=False
    )


def plot_disagreement_breakdown(joined: pd.DataFrame, tier: str, output_path) -> pd.DataFrame:
    data = _disagreement_breakdown_data(joined)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(data) + 1)))

    if data.empty:
        ax.text(0.5, 0.5, "No disagreements", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [f"Human {h.title()} → DeepSeek {c.title()}" for h, c in zip(data["human_label"], data["llm_classification"])]
        bars = ax.barh(labels[::-1], data["count"].values[::-1], color="#C44E52")
        for bar, value in zip(bars, data["count"].values[::-1]):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {int(value):,}", va="center", fontsize=9)
        ax.set_xlabel("Records")

    ax.set_title(f"Disagreement Breakdown -- DeepSeek ({tier}) Vs. Human")
    _add_caption(fig, "Every off-diagonal cell of the confusion matrix, ranked by count.", width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 6. Undeterminable-subset fallback comparison (forced-guess / RAG re-asks)
# ---------------------------------------------------------------------------


def compute_fallback_accuracy(fallback_events: pd.DataFrame, sample_df: pd.DataFrame, mode: str) -> dict:
    """`fallback_events`: `runner.classify_records(..., mode="forced_guess"|"rag")`'s output,
    joined against `sample_df`'s true `label`. parse_error rows excluded, same as every other
    accuracy figure in this module."""
    merged = fallback_events.merge(sample_df[["record_id", "label"]], on="record_id", how="inner")
    merged = merged[merged["classification"] != PARSE_ERROR]
    accuracy = float((merged["label"] == merged["classification"]).mean()) if len(merged) else float("nan")
    return {"mode": mode, "n": int(len(merged)), "accuracy": accuracy}


def plot_undeterminable_fallback_comparison(
    primary_agreement: dict, forced_guess: dict | None, rag: dict | None, output_path
) -> pd.DataFrame:
    """`primary_agreement`: one tier's `compute_agreement(...)` dict (its `undetermined_rate` is
    the baseline this chart contextualizes). `forced_guess`/`rag`: `compute_fallback_accuracy(...)`
    dicts, or None if that fallback wasn't run (e.g. DeepSeek has no search-tool support -- see
    STEPS_Progress.md Step 20's build-order §8b verification step)."""
    rows = [{"condition": "Primary Undetermined Rate", "value": primary_agreement["undetermined_rate"]}]
    if forced_guess is not None:
        rows.append({"condition": "Forced-Guess Accuracy\n(on undetermined subset)", "value": forced_guess["accuracy"]})
    if rag is not None:
        rows.append({"condition": "RAG-Assisted Accuracy\n(on undetermined subset)", "value": rag["accuracy"]})
    data = pd.DataFrame(rows)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#8172B3", "#55A868", "#4C72B0"][: len(data)]
    bars = ax.bar(data["condition"], data["value"], color=colors)
    _bar_value_labels(ax, bars, [v * 100 for v in data["value"]], fmt="{:.1f}%")
    ax.set_ylim(top=1.15)
    ax.set_ylabel("Rate")
    ax.set_title(f"Undeterminable Subset (n={primary_agreement['n_undetermined']:,}): Fallback Comparison")
    caption = (
        "Forced-guess/RAG accuracy is scored ONLY on the primary run's undetermined subset, "
        "against the human's real label -- meaningfully-above-chance forced-guess accuracy is "
        "evidence the model had a usable signal it was being falsely modest about; a materially "
        "higher RAG-assisted accuracy is evidence withholding search cost DeepSeek real accuracy "
        "on these specific hard cases."
    )
    _add_caption(fig, caption, width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 7. Step 20a: post-Cross-Curate-Resolve consensus -- did the human's FINAL decision end up
# agreeing with DeepSeek more often than the ORIGINAL label did? Built directly off
# cross_curate_resolution_events.csv, same as cohort_filters.py::build_disagreement_queue --
# deliberately no dependency on Step 20b's materialize step having run first.
# ---------------------------------------------------------------------------


def build_post_review_final_labels(sample: pd.DataFrame, resolution_events: pd.DataFrame) -> pd.DataFrame:
    """`sample[['record_id','label']]` renamed to `original_label`, plus `final_label`: the
    latest Cross Curate Resolve decision for the reviewed records, `original_label` unchanged for
    every record that was never in the disagreement queue at all."""
    result = sample[["record_id", "label"]].rename(columns={"label": "original_label"}).copy()
    result["final_label"] = result["original_label"]
    if resolution_events.empty:
        return result

    latest = resolution_events.sort_values("timestamp").groupby("record_id").last()
    result = result.set_index("record_id")
    common = latest.index.intersection(result.index)
    result.loc[common, "final_label"] = latest.loc[common, "decision"]
    return result.reset_index()


def compute_post_review_reversal_breakdown(
    sample: pd.DataFrame, resolution_events: pd.DataFrame, llm_events: pd.DataFrame, tier: str
) -> pd.DataFrame:
    """For every record where `tier`'s primary classification disagreed with the ORIGINAL human
    label (i.e. it was in the disagreement queue for this tier), buckets the post-review outcome:
    "Reversed to DeepSeek" (the final decision matches the tier's classification and differs from
    the original label), "Upheld Original" (the final decision matches the original label),
    "Neither" (the final decision is something else again -- e.g. the 1 undeterminable case)."""
    final_labels = build_post_review_final_labels(sample, resolution_events)
    tier_events = llm_events[(llm_events["mode"] == "primary") & (llm_events["model_tier"] == tier)]
    latest_tier = tier_events.sort_values("timestamp").groupby("record_id").last()
    joined = final_labels.merge(latest_tier[["classification"]], left_on="record_id", right_index=True, how="inner")
    disagreements = joined[joined["classification"] != joined["original_label"]].copy()

    def _bucket(row) -> str:
        if row["final_label"] == row["classification"] and row["final_label"] != row["original_label"]:
            return "Reversed to DeepSeek"
        if row["final_label"] == row["original_label"]:
            return "Upheld Original"
        return "Neither"

    disagreements["bucket"] = disagreements.apply(_bucket, axis=1)
    order = ["Reversed to DeepSeek", "Upheld Original", "Neither"]
    counts = disagreements["bucket"].value_counts().reindex(order, fill_value=0)
    return counts.to_frame(name="count")


def plot_post_review_reversal_breakdown(
    sample: pd.DataFrame, resolution_events: pd.DataFrame, llm_events: pd.DataFrame, tier: str, output_path
) -> pd.DataFrame:
    data = compute_post_review_reversal_breakdown(sample, resolution_events, llm_events, tier)
    total = int(data["count"].sum())

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#55A868", "#4C72B0", "#8C8C8C"]
    bars = ax.bar(data.index, data["count"], color=colors)
    _bar_value_labels(ax, bars, data["count"])
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Records")
    ax.set_title(f"Post-Review Outcome — DeepSeek ({tier}) Disagreements")
    caption = (
        f"n={total:,} records where {tier}'s primary classification disagreed with the original "
        "human label; outcome after the Cross Curate Resolve review's final human decision."
    )
    _add_caption(fig, caption, width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


def plot_original_vs_post_review_kappa(
    sample: pd.DataFrame, resolution_events: pd.DataFrame, llm_events: pd.DataFrame, output_path
) -> pd.DataFrame:
    """Grouped bar, per tier: kappa scored against the ORIGINAL label vs. kappa scored against the
    FINAL (post-review) label, over the same full sample population -- if post-review kappa is
    higher, the review moved the dataset closer to what DeepSeek was independently seeing."""
    final_labels = build_post_review_final_labels(sample, resolution_events)
    primary = llm_events[llm_events["mode"] == "primary"]

    rows = []
    for tier in sorted(primary["model_tier"].unique()):
        tier_events = primary[primary["model_tier"] == tier].sort_values("timestamp").groupby("record_id").last()
        original_joined = (
            final_labels[["record_id", "original_label"]]
            .rename(columns={"original_label": "label"})
            .merge(tier_events[["classification"]], left_on="record_id", right_index=True, how="inner")
        )
        final_joined = (
            final_labels[["record_id", "final_label"]]
            .rename(columns={"final_label": "label"})
            .merge(tier_events[["classification"]], left_on="record_id", right_index=True, how="inner")
        )
        original_agreement = compute_agreement(original_joined, tier)
        final_agreement = compute_agreement(final_joined, tier)
        rows.append(
            {
                "tier": tier,
                "kappa_original": original_agreement["kappa"],
                "kappa_post_review": final_agreement["kappa"],
            }
        )
    data = pd.DataFrame(rows)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(len(data))
    width = 0.35
    bars_original = ax.bar(
        [p - width / 2 for p in positions], data["kappa_original"], width, label="Original Label", color="#8172B3"
    )
    bars_final = ax.bar(
        [p + width / 2 for p in positions], data["kappa_post_review"], width,
        label="Post-Review Final Label", color="#55A868",
    )
    for bar, value in zip(bars_original, data["kappa_original"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars_final, data["kappa_post_review"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(data["tier"]))
    ax.set_ylim(top=1.15)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Kappa: Original Label Vs. Post-Review Final Label")
    ax.legend()
    caption = (
        "Post-review kappa substitutes the human's final Cross Curate Resolve decision for the "
        "original label wherever a review happened (~894/1000 records are unchanged either way) "
        "-- a higher post-review bar means the review moved the dataset closer to what DeepSeek "
        "was independently seeing, not the other way around."
    )
    _add_caption(fig, caption, width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 8. Step 20f: full-population re-run (Step 20e) profiling -- trusted-pool agreement at full
# scale, the EPMC "clear negative" candidate pool's confirmation rate, and whole-evaluated-
# population alignment against the truest available label. `criteria_hash`-aware throughout: the
# original 1,000-sample and the new full re-run were classified under two different CRITERIA.md
# versions (different criteria_sha256), so results are never blended into one number across that
# boundary -- every function below takes an explicit criteria_hash rather than assuming "latest
# event wins" is safe on its own.
# ---------------------------------------------------------------------------

_TRUSTED_LABEL_CONFIDENCE = ("human_curated", "registry_confirmed")


def build_population_run_joined(
    dataset: pd.DataFrame,
    llm_events: pd.DataFrame,
    tier: str,
    criteria_hash: str,
    label_confidences: tuple = _TRUSTED_LABEL_CONFIDENCE,
) -> pd.DataFrame:
    """The `joined` (record_id/label/classification) shape `compute_agreement()`/
    `plot_confusion_matrix_per_tier()`/`plot_disagreement_breakdown()` already expect, built from a
    real classify run (Step 20e) rather than the `second_curator_sample.csv` fixture -- restricted
    to one exact `criteria_hash` so results are never silently blended across a CRITERIA.md edit.
    `label_confidences` defaults to the trusted human-curated tier; pass `("heuristic_candidate",)`
    for the EPMC "clear negative" pool instead (see `build_candidate_pool_joined`, which wraps this
    for that case)."""
    primary = llm_events[
        (llm_events["model_tier"] == tier)
        & (llm_events["mode"] == "primary")
        & (llm_events["criteria_sha256"] == criteria_hash)
    ]
    latest = primary.sort_values("timestamp").groupby("record_id").last()["classification"]
    pool = dataset[
        dataset["label_confidence"].isin(label_confidences) & dataset["label"].isin(["positive", "negative"])
    ]
    joined = pool[["record_id", "label"]].merge(
        latest.rename("classification"), left_on="record_id", right_index=True, how="inner"
    )
    return joined.reset_index(drop=True)


def _source_batch(sources_cell) -> str:
    """Parses `canonical_dataset.csv`'s `sources` JSON to distinguish the two EPMC "clear negative"
    fetch batches -- Step 14's original `clear_negative_sampler_strong` vs. Step 19d's targeted
    follow-up `clear_negative_sampler_strong_filtered_v2` -- by substring match on the raw JSON
    text (cheap, and every relevant row's `source_name` is one of exactly these two literal strings
    by construction, see `ingest/clear_negative_sampler.py`)."""
    if not isinstance(sources_cell, str):
        return "unknown"
    if "clear_negative_sampler_strong_filtered_v2" in sources_cell:
        return "Step 19d (filtered_v2)"
    if "clear_negative_sampler_strong" in sources_cell:
        return "Step 14 (original)"
    return "unknown"


def build_candidate_pool_joined(
    dataset: pd.DataFrame, llm_events: pd.DataFrame, tier: str, criteria_hash: str
) -> pd.DataFrame:
    """The EPMC `heuristic_candidate` "clear negative" pool -- NEVER individually human-reviewed
    (`label == "negative"` is assumed by fetch-query design, not a verified per-record decision),
    so this is a diagnostic population, not a kappa-scoring one (see
    `compute_candidate_pool_confirmation`, below). Adds a `source_batch` column distinguishing the
    two fetch batches."""
    joined = build_population_run_joined(
        dataset, llm_events, tier, criteria_hash, label_confidences=("heuristic_candidate",)
    )
    sources_by_id = dataset.set_index("record_id")["sources"]
    joined["source_batch"] = joined["record_id"].map(lambda rid: _source_batch(sources_by_id.get(rid)))
    return joined


def compute_candidate_pool_confirmation(candidate_joined: pd.DataFrame) -> pd.DataFrame:
    """Per `source_batch`: how many of the never-individually-reviewed EPMC "clear negative"
    records did DeepSeek confirm negative vs. flag positive/undeterminable (parse_error excluded
    from the denominator, never coerced into a guess) -- the real, data-driven input to how far
    Gavin widens the live EPMC AI/ML search (see STEPS_Progress.md Step 20f)."""
    scored = candidate_joined[candidate_joined["classification"] != PARSE_ERROR]
    counts = pd.crosstab(scored["source_batch"], scored["classification"])
    for label in _LLM_LABELS:
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[[c for c in _LLM_LABELS if c in counts.columns]]
    totals = counts.sum(axis=1)
    result = counts.copy()
    result["n_total"] = totals
    for label in counts.columns:
        result[f"{label}_rate"] = counts[label] / totals
    return result


def plot_candidate_pool_confirmation(candidate_joined: pd.DataFrame, tier: str, output_path) -> pd.DataFrame:
    data = compute_candidate_pool_confirmation(candidate_joined)
    rate_cols = [f"{label}_rate" for label in _LLM_LABELS if f"{label}_rate" in data.columns]
    colors = {"negative_rate": "#55A868", "positive_rate": "#C44E52", "undeterminable_rate": "#8172B3"}

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = list(range(len(data)))
    width = 0.8 / max(len(rate_cols), 1)
    for i, col in enumerate(rate_cols):
        offsets = [p - 0.4 + width / 2 + i * width for p in positions]
        bars = ax.bar(
            offsets, data[col] * 100, width, label=col.replace("_rate", "").title(), color=colors.get(col, "#4C72B0")
        )
        for bar, value in zip(bars, data[col] * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{idx}\n(n={int(n):,})" for idx, n in zip(data.index, data["n_total"])])
    ax.set_ylim(top=115)
    ax.set_ylabel("% of batch")
    ax.set_title(f'EPMC "Clear Negative" Candidate Pool -- DeepSeek ({tier}) Confirmation Rate')
    ax.legend()
    _add_caption(
        fig,
        "These records were fetched by a query designed to exclude AI/ML terms and were never "
        "individually human-reviewed -- label='negative' is assumed by the fetch design, not "
        "verified per record. A high negative rate confirms the fetch design is clean; a "
        "meaningful positive/undeterminable rate flags records worth a manual look before trusting "
        "a wider EPMC search built the same way.",
        width=105,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data


def resolve_prior_criteria_hash(llm_events: pd.DataFrame, record_ids: set, tier: str, current_hash: str) -> str:
    """The criteria_sha256 an EARLIER classify run over `record_ids` actually used (e.g. Step 20's
    original 1,000-sample run, before Step 20c's CRITERIA.md edit) -- whichever hash, other than
    `current_hash`, has primary-mode events for this tier over these records. Raises if none found
    (nothing to compare against) or if more than one non-current hash exists (ambiguous -- pick
    explicitly rather than silently guessing which prior run to use)."""
    primary = llm_events[
        (llm_events["model_tier"] == tier)
        & (llm_events["mode"] == "primary")
        & (llm_events["record_id"].isin(record_ids))
        & (llm_events["criteria_sha256"] != current_hash)
    ]
    hashes = sorted(primary["criteria_sha256"].dropna().unique())
    if not hashes:
        raise ValueError(
            f"No classification events found for tier={tier!r} over this record set under any "
            "criteria_sha256 other than the current one -- nothing to compare against."
        )
    if len(hashes) > 1:
        raise ValueError(
            f"Multiple prior criteria_sha256 values found: {hashes!r} -- pick one explicitly "
            "rather than guessing which prior run to compare against."
        )
    return hashes[0]


def build_original_sample_vs_final_joined(
    sample: pd.DataFrame, resolution_events: pd.DataFrame, llm_events: pd.DataFrame, tier: str, criteria_hash: str
) -> pd.DataFrame:
    """The original evaluation sample, scored against the TRUEST available label -- the
    post-Cross-Curate-Resolve final decision (`build_post_review_final_labels`), not the original
    pre-review label -- joined against DeepSeek's classification under `criteria_hash` (the
    criteria version that sample was actually classified under; see `resolve_prior_criteria_hash`,
    above, to find it)."""
    final_labels = build_post_review_final_labels(sample, resolution_events)
    primary = llm_events[
        (llm_events["model_tier"] == tier)
        & (llm_events["mode"] == "primary")
        & (llm_events["criteria_sha256"] == criteria_hash)
    ]
    latest = primary.sort_values("timestamp").groupby("record_id").last()["classification"]
    joined = (
        final_labels[["record_id", "final_label"]]
        .rename(columns={"final_label": "label"})
        .merge(latest.rename("classification"), left_on="record_id", right_index=True, how="inner")
    )
    return joined.reset_index(drop=True)


def plot_whole_evaluated_population_comparison(
    original_joined: pd.DataFrame, new_joined: pd.DataFrame, tier: str, output_path
) -> pd.DataFrame:
    """Side-by-side kappa/accuracy bars: the original evaluation sample (scored against its truest,
    post-review final label, under the criteria it was actually classified under) vs. the new full
    re-run (scored against the current label, under the refined criteria) -- deliberately two
    separate bars, never pooled into one blended number, since the two groups were classified under
    different CRITERIA.md versions and scored against different notions of ground truth. Reuses
    `plot_kappa_accuracy_summary` unchanged -- its `agreements` dict just needs descriptive keys."""
    agreements = {
        f"Original {len(original_joined):,}\n(prior criteria, vs. final post-review label)": compute_agreement(
            original_joined, tier
        ),
        f"New {len(new_joined):,}\n(refined criteria, vs. current label)": compute_agreement(new_joined, tier),
    }
    return plot_kappa_accuracy_summary(agreements, output_path)


def plot_population_original_vs_post_review_kappa(
    population: pd.DataFrame, resolution_events: pd.DataFrame, llm_events: pd.DataFrame, output_path
) -> pd.DataFrame:
    """Same computation as `plot_original_vs_post_review_kappa`, generalized to any population
    DataFrame (record_id + label) rather than hardcoded to the original 1,000-sample -- for the
    full-population Cross Curate Resolve pass Step 20e's new disagreements go through. Kept as a
    separate function (not a parameter on the original) because that function's caption text is
    specific to the "~894/1000 records unchanged" framing, which would be actively wrong for a
    different population size."""
    final_labels = build_post_review_final_labels(population, resolution_events)
    primary = llm_events[llm_events["mode"] == "primary"]

    rows = []
    for tier in sorted(primary["model_tier"].unique()):
        tier_events = primary[primary["model_tier"] == tier].sort_values("timestamp").groupby("record_id").last()
        original_joined = (
            final_labels[["record_id", "original_label"]]
            .rename(columns={"original_label": "label"})
            .merge(tier_events[["classification"]], left_on="record_id", right_index=True, how="inner")
        )
        final_joined = (
            final_labels[["record_id", "final_label"]]
            .rename(columns={"final_label": "label"})
            .merge(tier_events[["classification"]], left_on="record_id", right_index=True, how="inner")
        )
        rows.append(
            {
                "tier": tier,
                "kappa_original": compute_agreement(original_joined, tier)["kappa"],
                "kappa_post_review": compute_agreement(final_joined, tier)["kappa"],
            }
        )
    data = pd.DataFrame(rows)
    n_changed = int((final_labels["original_label"] != final_labels["final_label"]).sum())
    n_total = int(len(final_labels))

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(len(data))
    width = 0.35
    bars_original = ax.bar(
        [p - width / 2 for p in positions], data["kappa_original"], width, label="Original Label", color="#8172B3"
    )
    bars_final = ax.bar(
        [p + width / 2 for p in positions], data["kappa_post_review"], width,
        label="Post-Review Final Label", color="#55A868",
    )
    for bar, value in zip(bars_original, data["kappa_original"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars_final, data["kappa_post_review"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(data["tier"]))
    ax.set_ylim(top=1.15)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Kappa: Original Label Vs. Post-Review Final Label (Full Population)")
    ax.legend()
    caption = (
        f"Post-review kappa substitutes the human's final Cross Curate Resolve decision for the "
        f"original label wherever a review happened ({n_changed:,}/{n_total:,} records changed) -- "
        "a higher post-review bar means the review moved the dataset closer to what DeepSeek was "
        "independently seeing, not the other way around."
    )
    _add_caption(fig, caption, width=105)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return data
