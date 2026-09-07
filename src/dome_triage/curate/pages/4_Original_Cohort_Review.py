"""Step 19: re-review of the pre-Streamlit-app manual curation cohort -- ~3,356 positive/negative
records curated before this Curate app existed (`DOME_Top_Curate`, `copilot_1012`, etc.), possibly
under a stricter/different standard than the now-consolidated criteria (see
`curation_criteria/CRITERIA.md`). This is a second, independent check of those original manual
calls -- explicitly NOT the DOME registry gold set (`dome_registry_231_gold`/`dome_registry_
222_gold`/`ebi_search_dome_api`), which is settled provenance-confirmed ground truth and out of
scope here (see `cohort_filters.py::build_original_cohort` for the exact filter).

**Default view is the FULL cohort, not a sample** -- opening this page shows every eligible record
(minus whatever's already been decided this pass), so browsing/filtering/curating works
immediately with no draw required. Sampling is an opt-in narrowing, not a prerequisite.

**One filter drives everything -- live, no button needed.** The "Original label (re-checking...)"
multiselect above the Sample section is the *only* label-scoping control on this page. Picking
"negative" immediately narrows `total()`/`remaining()` (whatever's currently in view -- full
cohort or a drawn sample), and is also what a fresh sample draws from -- "250" with "negative"
selected draws 250 from the negative-only population, not 250 mixed then filtered down afterward
(which could leave far fewer than 250 once narrowed). An earlier version had a second, separate
"Draw from" scope selector inside the Sample expander that only took effect on a button click --
selecting a scope there silently did nothing to the displayed remaining-count until you also
clicked Draw, which read as "the view isn't live-updating." Collapsed into one filter now.

**Sampling algorithm, documented here because it's the one thing this page does that isn't a
straightforward port of `1_Curate.py`'s pattern (see STEPS_Progress.md Step 19 for the same text,
kept in sync):**

1. The full cohort (~3,356 records) is loaded once per process and BM25-scored via the same
   bulk-match lookup every other page shares (`streamlit_helpers.get_original_cohort()`).
2. "Draw a fresh sample" runs `cohort_filters.sample_cohort()` against whatever the label filter
   above is currently showing (the full cohort, or just one label): a *stratified* sample within
   that scope (BM25 score band x journal bucket x year bucket --
   `sampling/stratified.py::build_strata`/`stratified_sample`, the exact pair Step 13's original
   queue used), seeded with `42 + draw_count` where `draw_count` starts at 0 and increments once
   per click within this browser session -- deterministic and reproducible (the same draw number
   always reproduces the same sample), but each successive draw is a genuinely different sample,
   not a repeat. Records already decided in `original_cohort_review_events.csv` are excluded
   before drawing, so redrawing never hands back something already reviewed this pass -- this
   exclusion applies whether browsing the full cohort or a drawn sample, since `CurationSession`
   itself also filters its queue against that same event log regardless of which population it
   was constructed from.
3. The drawn sample's record_ids are persisted in `st.session_state` (not redrawn on every
   rerun -- only on an explicit "Draw a fresh sample" click). Changing the label filter *after* a
   sample is drawn narrows that sample live, same as it narrows the full cohort -- it does not
   trigger a new draw. "Show full cohort again" clears the drawn sample and returns to the default
   full-cohort (label-filtered) view.
4. Presentation *order* (`Order:` radio below) is a separate, independent choice, applied to
   whatever's currently in view: "Random" reshuffles via `pandas.Series.sample(frac=1,
   random_state=...)` (NumPy's PCG64 generator) using the current `42 + draw_count` seed (so a
   given view's presentation order is also reproducible); "BM25 high-to-low"/"low-to-high" instead
   walks in score order, for a curator who wants to prioritize the highest- or lowest-confidence
   matches first rather than a random walk. Records with no BM25 score (~11% of this cohort -- it
   was never touched by the bulk AI/ML search that produced these scores) sort last either way.

**Review-term down-weighting.** A "Hide likely reviews / non-methods papers" checkbox does a
direct, case-insensitive keyword match of each record's title+abstract against the BM25
exclusionary lexicon's `non_methods_pubtype` terms (`keyword_lexicon_exclusionary.csv` --
"systematic review", "meta-analysis", "editorial", "commentary", "case report", etc.; 13 terms
confirmed live, the same list already curated for BM25 scoring, reused here rather than
duplicated -- see `cohort_filters.load_review_term_list`/`annotate_review_term_match`). Off by
default; when checked, filters OUT any matching record from the current view. This is a plain
keyword match, not a score -- the point is an explainable "this record's own text literally names
a review/non-methods pub type" signal, so a curator re-checking negatives can skip re-verifying
the ~11% (425/3,356 confirmed live, 358 of them already-negative) that are almost certainly
already correctly negative, and spend review time on the harder/ambiguous cases instead.

Decisions write to `data/processed/original_cohort_review_events.csv` -- a separate event log from
the main `curation_events.csv`, so this second-check pass stays distinguishable in the audit trail
(same `curate materialize-original-cohort-review` CLI command folds it into
`canonical_dataset.csv`, reusing `materialize_events` unchanged -- a decision that contradicts the
record's trusted original label becomes `label="conflict"`, exactly like every other materialize
path in this app; see AGENTS.md's "human curation is never bypassed" rule).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dome_triage.config import resolve_path
from dome_triage.curate.streamlit_helpers import (
    draw_original_cohort_sample,
    get_config,
    get_original_cohort,
    get_original_cohort_session,
    get_review_term_list,
    get_youden_threshold,
)

st.set_page_config(page_title="Original Cohort Review", layout="wide")
st.title("Original Cohort Review")
st.caption(
    "Second-check re-review of the manual pre-app curation set (DOME_Top_Curate, copilot_1012, "
    "etc.) -- NOT the DOME registry gold set, which is out of scope (see this page's module "
    "docstring for the exact provenance filter)."
)

if not st.session_state.get("curator_name"):
    st.warning("Set your curator name on the Home page first.")
    st.stop()

cfg = get_config()
if not cfg.path("canonical_dataset").exists():
    st.info("No canonical_dataset.csv yet -- run the ingest/dedupe pipeline first.")
    st.stop()

cohort = get_original_cohort()
events_path = resolve_path(cfg.pipeline["curation"]["original_cohort_review_events"])
decided_ids: set[str] = set()
if events_path.exists():
    decided_ids = set(pd.read_csv(events_path, usecols=["record_id"], dtype=str)["record_id"])

n_pos = int((cohort["label"] == "positive").sum())
n_neg = int((cohort["label"] == "negative").sum())
n_scored = int(cohort["bulk_match_score"].notna().sum())
st.caption(
    f"Full cohort: **{len(cohort):,}** records ({n_pos:,} originally positive / {n_neg:,} "
    f"originally negative) -- {n_scored:,} ({n_scored / len(cohort) * 100:.0f}%) have a BM25 "
    f"score. **{len(decided_ids):,}** already re-reviewed via this page so far (all-time, across "
    "every draw)."
)

if cohort.empty:
    st.success("Nothing left in the original cohort to re-review -- every eligible record has "
               "either already been re-reviewed here or reviewed via the main Curate page.")
    st.stop()

# -------------------------------------------------------------------------------------------
# Filters -- label (the ORIGINAL manual decision being re-checked) and presentation order. This
# is deliberately the FIRST thing computed, and the ONE control that determines label scope --
# it applies LIVE to whatever's currently in view (full cohort or a drawn sample), and it's also
# what "Draw a fresh sample" below draws from. An earlier version had a second, separate "Draw
# from" scope selector in the Sample section that only took effect on a button click -- picking
# "negative" there silently did nothing to the displayed remaining-count until you also clicked
# Draw, which read as "the view isn't live-updating". Collapsing to this one filter fixes that:
# selecting a label here immediately changes total()/remaining() below, no button required.
# -------------------------------------------------------------------------------------------
filter_col, order_col = st.columns(2)
label_filter = filter_col.multiselect(
    "Original label (re-checking...)", options=["positive", "negative"], default=[],
    help="Leave empty to review both. Filters LIVE on the ORIGINAL pre-app manual label -- e.g. "
    "picking 'negative' immediately narrows the remaining count below to just originally-negative "
    "records, and is also what a fresh sample draw pulls from (e.g. '250' with 'negative' selected "
    "draws 250 from the negative-only population, not 250 mixed then filtered down).",
)
order_label = order_col.radio(
    "Order",
    options=["Random (seeded)", "BM25 score: high to low", "BM25 score: low to high"],
    horizontal=True,
)
order = (
    "random" if order_label.startswith("Random")
    else "bm25_desc" if "high to low" in order_label
    else "bm25_asc"
)

filtered_cohort = cohort[cohort["label"].isin(label_filter)] if label_filter else cohort

# -------------------------------------------------------------------------------------------
# Review-term down-weighting -- direct keyword match against the BM25 exclusionary lexicon's
# review/non-methods terms (systematic review, meta-analysis, editorial, etc. -- the "big list"
# already curated in Step 9-11's lexicon work, reused here via cohort_filters.py, not
# duplicated). Purpose: skip re-verifying negatives that are almost certainly correctly negative
# because their own title/abstract literally names a review/non-methods pub type -- an opt-in
# toggle, off by default, so nothing is hidden unless asked for.
# -------------------------------------------------------------------------------------------
review_terms = get_review_term_list()
n_review_matches_in_view = int(filtered_cohort["matches_review_term"].sum())
review_terms_help = (
    "Direct keyword match against the BM25 exclusionary lexicon's review/non-methods terms: "
    + ", ".join(review_terms)
    if review_terms
    else "No exclusionary lexicon found yet -- run `keywords build-lexicon`/"
    "`keywords seed-additional-terms` first."
)
hide_review_matches = st.checkbox(
    f"Hide likely reviews / non-methods papers ({n_review_matches_in_view:,} matched in the "
    "current view)",
    value=False,
    help=review_terms_help,
)
if hide_review_matches:
    filtered_cohort = filtered_cohort[~filtered_cohort["matches_review_term"]]

# -------------------------------------------------------------------------------------------
# View: defaults to the FULL (label-filtered) cohort -- cohort_sample_ids stays None until an
# explicit draw, so sampling is opt-in, not a prerequisite for browsing/curating. A drawn sample
# is persisted in session_state; only a real "Draw a fresh sample" click redraws (see module
# docstring for the exact seeding algorithm). Changing the label filter above while a sample is
# active does NOT redraw the sample -- it narrows the already-drawn sample live, same as it
# narrows the full cohort; draw a fresh sample to get a new draw from the new filter's scope.
# -------------------------------------------------------------------------------------------
st.session_state.setdefault("cohort_draw_count", 0)
st.session_state.setdefault("cohort_sample_size", min(250, len(cohort)))
st.session_state.setdefault("cohort_sample_ids", None)
current_seed = 42 + st.session_state["cohort_draw_count"]

viewing_sample = st.session_state["cohort_sample_ids"] is not None
with st.expander("Sample", expanded=not viewing_sample):
    scope_pool_size = len(filtered_cohort)
    scope_desc = f"{', '.join(label_filter)}" if label_filter else "positive + negative"
    if viewing_sample:
        st.caption(
            f"Viewing a drawn sample of **{len(st.session_state['cohort_sample_ids']):,}** "
            f"({st.session_state['cohort_sample_scope_desc']}) -- draw "
            f"#{st.session_state['cohort_draw_count']}, seed {st.session_state['cohort_draw_seed']}. "
            f"The label filter above is now narrowing that sample, not the full cohort."
        )
    else:
        st.caption(f"Viewing **{len(filtered_cohort):,}** records ({scope_desc}).")

    sample_size = st.number_input(
        f"Sample size (drawn fresh from the {scope_desc} population above, stratified by BM25 "
        "score band x journal x year)",
        min_value=1,
        max_value=max(scope_pool_size, 1),
        value=min(st.session_state["cohort_sample_size"], max(scope_pool_size, 1)),
        step=10,
    )
    draw_col1, draw_col2, draw_col3 = st.columns([1, 1, 2])
    if draw_col1.button("Draw a fresh sample", type="primary"):
        st.session_state["cohort_sample_size"] = sample_size
        drawn = draw_original_cohort_sample(
            filtered_cohort, int(sample_size), current_seed, exclude_ids=decided_ids
        )
        st.session_state["cohort_sample_ids"] = drawn["record_id"].tolist()
        st.session_state["cohort_sample_scope_desc"] = scope_desc
        st.session_state["cohort_draw_seed"] = current_seed
        st.session_state["cohort_draw_count"] += 1
        st.rerun()
    if viewing_sample and draw_col2.button("Show full cohort again"):
        st.session_state["cohort_sample_ids"] = None
        st.rerun()

if viewing_sample:
    view_df = filtered_cohort[filtered_cohort["record_id"].isin(set(st.session_state["cohort_sample_ids"]))]
else:
    view_df = filtered_cohort

session = get_original_cohort_session(view_df, order=order, order_seed=current_seed)

with st.sidebar.expander("Diversity tracker (this review pass)", expanded=True):
    st.caption("Year and journal coverage of records re-decided so far, scoped to the current view "
               "(full cohort or drawn sample).")
    stats = session.diversity_stats()
    st.metric(
        "Journal coverage (re-decided)",
        f"{stats['n_journals_covered']} / {stats['n_journals_total']}",
        f"{stats['journal_coverage_pct']:.1f}%",
    )
    if not stats["per_year_counts"].empty:
        st.caption("Re-decided records by year")
        st.bar_chart(stats["per_year_counts"])
    if not stats["per_journal_counts"].empty:
        st.caption("Re-decided records by journal")
        st.dataframe(
            stats["per_journal_counts"]
            .sort_values(by=list(stats["per_journal_counts"].columns), ascending=False)
            .head(20)
        )

nav_prog, nav_back, nav_fwd = st.columns([6, 1, 1])
nav_prog.caption(f"{session.remaining()} of {session.total()} remaining (this view)")
nav_back.button(
    "< Back", disabled=not session.can_go_back(), on_click=session.go_back, use_container_width=True
)
nav_fwd.button(
    "Forward >",
    disabled=not session.can_go_forward(),
    on_click=session.go_forward,
    use_container_width=True,
)

record = session.current_record()
if record is None:
    st.success("No records left in this view to re-review. Widen/clear the label filter, draw a "
               "different sample, or click 'Show full cohort again' above.")
    st.stop()

prior_decision = session.current_record_prior_decision()
if prior_decision:
    st.info(f"You already marked this record **{prior_decision}** this session. Reviewing again.")

# ---------------------------------------------------------------------------------------------
# Paper display
# ---------------------------------------------------------------------------------------------
st.markdown(f"### {record.get('title') or '(no title)'}", unsafe_allow_html=True)
if bool(record.get("matches_review_term")):
    st.caption("Matches a review/non-methods term (see the 'Hide likely reviews' checkbox above).")


def _display_year(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "(unknown)"


original_label = record.get("label")
st.warning(
    f"**Original manual label: {str(original_label).upper()}** -- this is a second, independent "
    "check of that call. Submitting a different decision below does not silently overwrite it: "
    "`curate materialize-original-cohort-review` flags any contradiction as `label=\"conflict\"` "
    "for a final human tie-breaker, exactly like every other materialize path in this app."
)

bm25_score = record.get("bulk_match_score")
has_bm25 = bm25_score not in (None, "", "nan") and str(bm25_score) != "nan"

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.markdown(f"**Journal:** {record.get('journal') or '(unknown)'}", unsafe_allow_html=True)
meta_col2.markdown(f"**Year:** {_display_year(record.get('year'))}")
if has_bm25:
    threshold = get_youden_threshold()
    threshold_text = (
        f"Validated Youden threshold: {threshold:.1f} (Step 11's bake-off). Higher score = "
        "stronger lexicon match against the current AI/ML criteria -- a ranking signal to help "
        "triage, not a verdict."
        if threshold is not None
        else "No validated classification threshold found yet."
    )
    meta_col3.metric("BM25 match score", f"{float(bm25_score):.1f}", help=threshold_text)
else:
    meta_col3.caption("No BM25 score (never matched by the bulk AI/ML search).")

st.markdown(record.get("abstract") or "*(no abstract available)*", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------
# Decision -- same P/N/U/S immediate-submit pattern as 1_Curate.py (see that page's module
# docstring for why on_click callbacks, not a post-hoc `if button:` check).
# ---------------------------------------------------------------------------------------------


def _submit(decision: str):
    def _callback():
        notes = st.session_state.get("cohort_notes", "")
        session.record_decision(decision, notes=notes)
        st.session_state["cohort_notes"] = ""

    return _callback


btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
btn_col1.button("Positive (P)", on_click=_submit("positive"), use_container_width=True, type="primary")
btn_col2.button("Negative (N)", on_click=_submit("negative"), use_container_width=True)
btn_col3.button("Undeterminable (U)", on_click=_submit("undeterminable"), use_container_width=True)
btn_col4.button("Skip (S)", on_click=_submit("skipped"), use_container_width=True)

st.text_area("Notes (optional)", key="cohort_notes")
