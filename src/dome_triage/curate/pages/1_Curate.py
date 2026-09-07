"""Main Positive/Negative/Undeterminable/Skip curation queue -- ports curation.ipynb's
CurationInterface interaction pattern (see curate/state.py) into a Streamlit page.

Keyboard-first by design (P/N/U/S submit and advance immediately -- no staged "pick a decision,
then submit" step, since that's slower for the rapid-review workload this page exists for).
Decision buttons use `on_click` callbacks (not a post-hoc `if button:` check) specifically because
that's the only Streamlit-supported way to clear the Notes widget for the next record without
hitting "cannot modify a widget after it's instantiated" -- callbacks run in a phase before the
widget is re-instantiated on the next rerun. The keyboard-shortcut script below drives these same
buttons by simulating a real DOM click, so it doesn't need its own Python-side handling at all.
"""

from __future__ import annotations

import json

import streamlit as st
import yaml

from dome_triage.config import resolve_path
from dome_triage.curate.streamlit_helpers import (
    build_probe_session,
    get_bulk_pool_score_bounds,
    get_filter_options,
    get_session,
    get_youden_threshold,
)

st.set_page_config(page_title="Curate", layout="wide")
st.title("Curate")

if not st.session_state.get("curator_name"):
    st.warning("Set your curator name on the Home page first.")
    st.stop()

# ---------------------------------------------------------------------------------------------
# Queue source -- outside the Filters expander, always visible, since it changes what "Filters"
# even means below (score band vs. a raw score figure). "Curation queue" is Step 13's
# pre-stratified ~2-3k sample already merged into canonical_dataset.csv; "Full bulk pool" is the
# entire ~745k-record Europe PMC AI/ML match (bulk_candidates_scored.csv), ranked by BM25 score --
# not confined to whatever Step 13 happened to sample. A decision made from the bulk pool on a
# record never seen by canonical_dataset.csv is still safe: `curate materialize` (Step 16) inserts
# a new row for it rather than dropping the decision (see state.py's `_insert_missing_records_
# from_bulk_pool`).
# ---------------------------------------------------------------------------------------------
queue_source_label = st.radio(
    "Queue source",
    options=["Curation queue (Step 13 stratified sample)", "Full AI/ML bulk pool (~745k, ranked by BM25)"],
    horizontal=True,
    help="The stratified queue is the diverse, size-capped sample Step 13 already drew and merged "
    "into canonical_dataset.csv. The full bulk pool is every one of the ~745k Europe PMC records "
    "the AI/ML search matched, ranked by BM25 score highest-first -- pick this to browse/curate "
    "beyond the stratified sample, e.g. starting from a specific score figure downward.",
)
queue_source = "bulk_pool" if queue_source_label.startswith("Full AI/ML") else "canonical"

# ---------------------------------------------------------------------------------------------
# Filters -- collapsed by default so the paper under review stays the focus; state persists once
# you collapse it back down yourself.
# ---------------------------------------------------------------------------------------------
with st.expander("Filters", expanded=False):
    toggle_col1, toggle_col2 = st.columns(2)
    already_curated_mode = toggle_col1.radio(
        "Already-curated records (from the ORIGINAL curation effort)",
        options=["Hide (default)", "Include in queue", "Only show these (re-check)"],
        index=0,
        help="'Hide': settled ground truth -- DOME_Top_Curate, the DOME registry, or a decision "
        "you already materialized through this app -- stays out of the queue (the default). "
        "'Include in queue': mix already-curated records in alongside not-yet-curated ones. "
        "'Only show these': filter the queue down to *just* the already-curated records, for "
        "deliberately revisiting and possibly changing past decisions.",
    )
    include_already_labeled = already_curated_mode != "Hide (default)"
    only_already_labeled = already_curated_mode == "Only show these (re-check)"
    require_pmcid = toggle_col2.checkbox(
        "Only show records with full text available (has PMCID)", value=False
    )

    classification = st.multiselect(
        "BM25 classification (Step 12)", options=["positive", "negative"]
    )
    if queue_source == "canonical":
        needs_screening_only = st.checkbox(
            "Only show clear-negative candidates flagged for extra scrutiny (Step 14b)",
            value=False,
            help="A 'clear negative' (fetched specifically for NOT mentioning AI/ML terms) that "
            "still scored above the validated lexicon threshold -- worth a closer look before "
            "trusting it as a genuine negative. Not applicable to the full bulk pool (Step 14b "
            "only screens the clear-negative candidates, not the AI/ML-matched pool).",
        )
    else:
        needs_screening_only = False

    # Probe session: same base filters as above, but never touches get_session()'s cache slot --
    # only used to compute real score-range/curated-so-far numbers below, and the year slider's
    # bounds. queue_source-aware -- see build_probe_session()'s docstring.
    probe = build_probe_session(
        include_already_labeled=include_already_labeled,
        only_already_labeled=only_already_labeled,
        require_pmcid=require_pmcid,
        classification=classification or None,
        needs_screening_only=needs_screening_only,
        queue_source=queue_source,
    )

    if queue_source == "canonical":
        band_summary = probe.score_band_summary()
        _n_bands = len(band_summary)

        def _format_band(band_idx: int) -> str:
            info = next((b for b in band_summary if b["band"] == band_idx), None)
            if info is None:
                return f"Q{band_idx + 1}"
            extreme = " (lowest)" if band_idx == 0 else " (highest)" if band_idx == _n_bands - 1 else ""
            return (
                f"Q{band_idx + 1}{extreme}: BM25 {info['min_score']:.1f}-{info['max_score']:.1f} "
                f"-- {info['confirmed']}/{info['total']} curated"
            )

        score_band = st.multiselect(
            "Match-score band (BM25)",
            options=[b["band"] for b in band_summary],
            format_func=_format_band,
            help="Quartiles of match_score__bm25 (Q1 = lowest scores, Q4 = highest), computed "
            "fresh over whatever the other filters above currently leave in the queue -- not "
            "replaying Step 13's original 745k-pool-wide bands, which would be less useful now "
            "that the queue is already a small, pre-stratified subset. 'X/Y curated' = how many "
            "of that band's records already have a confirmed decision, out of the band's total. "
            "See STEPS_Progress.md's Step 13 section for exactly how these bands, the journal "
            "bucketing, and the per-stratum random sampling were computed.",
        )
        min_score = max_score = None
    else:
        # Full bulk pool: no quartile-band concept (§ROADMAP -- the whole point is not being
        # confined to Step 13's bands) -- instead, a raw score figure to browse from, always
        # ranked highest-to-lowest (CurationSession's sort_by_score_desc=True, set in
        # streamlit_helpers._bulk_pool_session_kwargs), with the Youden threshold shown explicitly
        # so it's clear where the validated bake-off found match quality starts to drop off.
        score_band = None
        _pool_bounds = get_bulk_pool_score_bounds()
        _pool_min, _pool_max = _pool_bounds if _pool_bounds else (0.0, 1000.0)
        _youden = get_youden_threshold()

        score_col1, score_col2 = st.columns(2)
        max_score = score_col1.number_input(
            "Start browsing at BM25 score ≤",
            min_value=_pool_min,
            max_value=_pool_max,
            value=_pool_max,
            step=1.0,
            help=f"The full bulk pool's real score range is {_pool_min:.1f} to {_pool_max:.1f}. "
            "Records are always shown highest-scoring first, so this is exactly where browsing "
            "starts -- lower it to jump straight to a specific point in the ranking.",
        )
        min_score = score_col2.number_input(
            "...down to BM25 score ≥",
            min_value=_pool_min,
            max_value=_pool_max,
            value=_pool_min,
            step=1.0,
            help="Leave at the pool minimum to keep browsing all the way to the bottom of the "
            "ranking; raise it to stop the queue above a floor.",
        )
        if min_score > max_score:
            st.warning("Minimum score is above the maximum -- no records will match. Adjust one of the two.")
        if _youden is not None:
            in_range = min_score <= _youden <= max_score
            st.caption(
                f"**Validated Youden threshold: {_youden:.1f}** (Step 11's bake-off, bm25 + "
                "exclusionary lexicon) -- Step 12 classified records scoring at or above this as "
                "positive, below it as negative. This is where the lexicon match starts to look "
                "unreliable in aggregate; it's a signpost for your own judgement, not a cutoff "
                "enforced anywhere in this view."
                + (
                    ""
                    if in_range
                    else " (outside your current min/max range above -- widen the range to see it.)"
                )
            )

    _filter_options = get_filter_options()
    journals = st.multiselect(
        "Journal (type to search -- every journal in the dataset, not just a top-N shortlist)",
        options=_filter_options["journals"],
    )

    # Year bounds come from the probe (the currently-filtered population), not the whole dataset
    # -- e.g. once you've filtered to classification=positive, the slider should only span years
    # actually present among positive-classified records, not the full corpus's range.
    _year_min, _year_max = probe.year_bounds() or (_filter_options["year_min"], _filter_options["year_max"])
    if _year_min == _year_max:
        # st.slider requires min_value < max_value strictly -- a real, reachable case once other
        # filters (journal/classification/score range) narrow the population to a single year,
        # not just a test-fixture artifact. Show the one year directly instead of a slider.
        st.caption(f"Year: **{_year_min}** (only one year present in the current filter)")
        year_range = (_year_min, _year_max)
    else:
        year_range = st.slider(
            "Year range (bounds reflect the filters above, not the whole dataset)",
            min_value=_year_min,
            max_value=_year_max,
            value=(_year_min, _year_max),
            help="Collapse both handles to the same year to pick a single year.",
        )

session = get_session(
    include_already_labeled=include_already_labeled,
    only_already_labeled=only_already_labeled,
    require_pmcid=require_pmcid,
    score_band=score_band or None,
    journals=journals or None,
    year_range=year_range if year_range != (_year_min, _year_max) else None,
    classification=classification or None,
    needs_screening_only=needs_screening_only,
    queue_source=queue_source,
    min_score=min_score,
    max_score=max_score,
)

with st.sidebar.expander("Diversity tracker", expanded=False):
    stats = session.diversity_stats()
    st.metric(
        "Journal coverage (confirmed pos/neg)",
        f"{stats['n_journals_covered']} / {stats['n_journals_total']}",
        f"{stats['journal_coverage_pct']:.1f}%",
    )
    if not stats["per_year_counts"].empty:
        st.caption("Positive/negative decisions by year")
        st.bar_chart(stats["per_year_counts"])
    if not stats["per_journal_counts"].empty:
        st.caption("Positive/negative decisions by journal (top by volume)")
        st.dataframe(
            stats["per_journal_counts"]
            .sort_values(by=list(stats["per_journal_counts"].columns), ascending=False)
            .head(20)
        )

# ---------------------------------------------------------------------------------------------
# Progress + navigation. Uses `on_click` rather than `if button:` + `st.rerun()`, for two separate
# reasons that both matter:
#   - No st.rerun(): Streamlit already reruns the script on any widget click, so an explicit one
#     forces a *second* full execution (re-reading both CSVs, rebuilding the scored pool) for zero
#     gain -- measured live as an exact doubling of the page's wall time.
#   - on_click, not `if button:`: callbacks run *before* the script body, so the move is already
#     applied when these buttons compute their own `disabled` state below. With `if button:` the
#     move happens after they've rendered, leaving both buttons a full interaction stale (clicking
#     Forward off position 0 correctly showed the next paper but left "< Back" greyed out).
# ---------------------------------------------------------------------------------------------
nav_prog, nav_back, nav_fwd = st.columns([6, 1, 1])
nav_prog.caption(f"{session.remaining()} of {session.total()} remaining (this filtered view)")
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
    st.success("No records left to curate.")
    st.stop()

prior_decision = session.current_record_prior_decision()
if prior_decision:
    st.info(f"You already marked this record **{prior_decision}** this session. Reviewing again.")

if journals and len(journals) == 1:
    j = journals[0]
    per_journal = stats["per_journal_counts"]
    if j in per_journal.index:
        pos = int(per_journal.loc[j].get("positive", 0))
        neg = int(per_journal.loc[j].get("negative", 0))
        st.caption(f"**{j}** so far: {pos} positive / {neg} negative confirmed")

needs_screening = bool(record.get("needs_screening"))
if needs_screening:
    bulk_score = record.get("bulk_match_score")
    score_text = f"{float(bulk_score):.1f}" if bulk_score not in (None, "") else "?"
    st.warning(
        f"This clear-negative candidate scored {score_text} against the lexicon despite the "
        "AI/ML exclusion query used to fetch it -- double-check before confirming negative."
    )

# ---------------------------------------------------------------------------------------------
# Paper display
# ---------------------------------------------------------------------------------------------
st.markdown(f"### {record.get('title') or '(no title)'}", unsafe_allow_html=True)

def _display_year(value) -> str:
    """Real years in canonical_dataset.csv are string-formatted floats ("2025.0") -- a pre-existing
    artifact of the consolidate pipeline's CSV round-tripping (see sampling/stratified.py, which
    handles the same thing). Rendering that raw put "Year: 2025.0" on every single paper."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "(unknown)"


bm25_score = record.get("bulk_match_score")
bm25_classification = record.get("bulk_match_classification")
has_bm25 = bm25_score not in (None, "", "nan") and str(bm25_score) != "nan"

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.markdown(f"**Journal:** {record.get('journal') or '(unknown)'}", unsafe_allow_html=True)
meta_col2.markdown(f"**Year:** {_display_year(record.get('year'))}")
if has_bm25:
    threshold = get_youden_threshold()
    threshold_text = (
        f"Classified **{bm25_classification or '(unscored)'}** because the score is "
        f"{'>=' if bm25_classification == 'positive' else '<'} the validated Youden threshold "
        f"of {threshold:.1f} (Step 11's bake-off, bm25 + exclusionary lexicon). Higher score = "
        "stronger lexicon match; this is a ranking signal to help triage, not a verdict -- your "
        "decision is what actually counts."
        if threshold is not None
        else "No validated classification threshold found yet (run `keywords scoring-bakeoff`)."
    )
    meta_col3.metric(
        "BM25 match score",
        f"{float(bm25_score):.1f}  ({bm25_classification or 'unscored'})",
        help=threshold_text,
    )

mesh_raw = record.get("mesh_headings")
if isinstance(mesh_raw, str) and mesh_raw.strip() not in ("", "[]"):
    try:
        mesh_list = json.loads(mesh_raw)
    except json.JSONDecodeError:
        mesh_list = []
    if mesh_list:
        st.markdown(f"**MeSH terms:** {', '.join(mesh_list)}")

# No fixed-height scrollable container here -- the whole abstract must be visible at a glance for
# fast curation, not require scrolling to read the end before deciding.
st.markdown(record.get("abstract") or "*(no abstract available)*", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------
# Decision -- P/N/U/S all submit immediately and advance (see module docstring for why there's no
# separate staged-decision step). Notes/optional feature widgets are read at submit time via
# their session_state keys, not through an st.form.
#
# Select-type features (currently just close_negative_reason) render as a row of small, numbered,
# hoverable toggle buttons -- visually separated above the P/N/U/S row, not a dropdown -- so a
# reason can be pre-picked (by click or a 1-9 keyboard shortcut) *before* deciding, without itself
# submitting anything. Clicking a decision button (or its P/N/U/S key) is still the only thing
# that calls record_decision(); a select toggle only ever writes to session_state. Numbering is
# global across every select-type feature (not restarted per feature), so a future feature #2
# never collides with feature #1's digit shortcuts -- capped at 9 keys, matching keyboards.
# ---------------------------------------------------------------------------------------------
_features_path = resolve_path("configs/curation_features.yaml")
_features_config = (
    yaml.safe_load(_features_path.read_text()).get("features", [])
    if _features_path.exists()
    else []
)

def _toggle_feature_option(state_key: str, option: str):
    def _callback():
        st.session_state[state_key] = "" if st.session_state.get(state_key) == option else option

    return _callback


_digit_shortcuts: dict[str, str] = {}  # "1".."9" -> exact button label, for the keyboard script below
_next_digit = 1
for feature in _features_config:
    state_key = f"curate_feat_{feature['key']}"
    if feature["type"] == "bool":
        st.checkbox(feature["label"], key=state_key)
        continue
    if feature["type"] != "select":
        continue

    st.session_state.setdefault(state_key, "")
    descriptions = feature.get("descriptions", {})
    st.caption(feature["label"])
    option_cols = st.columns(len(feature["options"]))
    for i, option in enumerate(feature["options"]):
        has_shortcut = _next_digit <= 9
        label = f"{_next_digit}: {option}" if has_shortcut else option
        option_cols[i].button(
            label,
            key=f"{state_key}_opt_{i}",
            on_click=_toggle_feature_option(state_key, option),
            type="primary" if st.session_state.get(state_key) == option else "secondary",
            help=descriptions.get(option, option),
            use_container_width=True,
        )
        if has_shortcut:
            _digit_shortcuts[str(_next_digit)] = label
            _next_digit += 1


def _submit(decision: str):
    def _callback():
        features = {}
        for feature in _features_config:
            state_key = f"curate_feat_{feature['key']}"
            value = st.session_state.get(state_key)
            shown_when = feature.get("shown_when_decision")
            # Only attach this feature's value to the decision it's actually scoped to (e.g.
            # close_negative_reason declares shown_when_decision: negative) -- otherwise a reason
            # selected before accidentally clicking Positive would attach to the wrong decision.
            # A feature with no shown_when_decision is unscoped and always eligible, unchanged.
            if value not in (None, "", False) and shown_when in (None, decision):
                features[feature["key"]] = value
            st.session_state[state_key] = "" if feature["type"] == "select" else False
        notes = st.session_state.get("curate_notes", "")
        session.record_decision(decision, notes=notes, features=features or None)
        st.session_state["curate_notes"] = ""

    return _callback


btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
btn_col1.button(
    "Positive (P)", on_click=_submit("positive"), use_container_width=True, type="primary"
)
btn_col2.button("Negative (N)", on_click=_submit("negative"), use_container_width=True)
btn_col3.button("Undeterminable (U)", on_click=_submit("undeterminable"), use_container_width=True)
btn_col4.button("Skip (S)", on_click=_submit("skipped"), use_container_width=True)

# Last on the page, deliberately -- if Notes sits above the decision row, it's easy to leave your
# cursor focused inside it after jotting something down, and the keyboard-shortcut script below
# intentionally ignores every key while a text input/textarea has focus (so typing "n" in a note
# types "n", it doesn't submit Negative). Notes' value is read from session_state at submit time
# regardless of where it sits visually, so moving it here doesn't change when a note is captured --
# it just keeps focus off a text field during normal keyboard-driven review.
st.text_area("Notes (optional)", key="curate_notes")

# ---------------------------------------------------------------------------------------------
# Keyboard shortcuts: P/N/U/S trigger the matching decision button, 1-9 toggle the matching
# feature-option button (see above -- selecting one this way does NOT submit anything, exactly
# like clicking it), all by simulating a real click on the underlying DOM element (this is the
# only way to drive a server-side on_click callback from client-side JS -- there's no direct
# Python hook to bind to). Two real, live bugs fixed here (both found by driving this page with an
# actual browser, not just Streamlit's AppTest, which never executes real JS/DOM at all):
#
#   1. "Shortcuts worked once, then stopped after touching a filter" -- adjusting a number_input
#      or the journal multiselect's search box leaves the browser's keyboard focus sitting inside
#      that <input> even after you're done with it. The isTyping guard below then correctly (by
#      design -- typing "n" in Notes must type "n") swallows every subsequent shortcut key, but
#      silently -- there was no way to tell "I'm in an input" from "this is just broken". Fixed by
#      blurring any stray-focused <input> (never TEXTAREA -- Notes must stay interruptible-free)
#      at the top of *every* rerun, not just once: st.iframe re-runs this whole script on every
#      Streamlit rerun (confirmed live), so this check runs fresh every time without needing its
#      own guard.
#   2. No visual confirmation a keypress was even received -- Streamlit's own rerun can take
#      seconds (the full bulk pool, especially), during which nothing on screen changes. Fixed
#      with `showFeedback()`, a small toast appended directly to `doc.body` (a sibling of
#      Streamlit's own React root, so Streamlit's re-renders never touch or clear it) that flashes
#      the matched action immediately -- independent of, and faster than, Streamlit's round trip.
#
# The keydown *listener* itself is still attached exactly once per page lifetime, guarded on a
# flag stashed on `window.parent` (the actual browser tab, not this component's own iframe) --
# without that guard, each rerun would stack another duplicate listener on the parent document,
# eventually firing one keypress N times. `_digit_shortcuts` is built in Python above from
# whatever `configs/curation_features.yaml` currently declares -- extending it with a 6th (or
# 10th) option needs no JS change here.
# ---------------------------------------------------------------------------------------------
st.iframe(
    """
    <script>
    (function() {
        const doc = window.parent.document;

        const stuckFocus = doc.activeElement;
        if (stuckFocus && stuckFocus.tagName === 'INPUT') { stuckFocus.blur(); }

        if (window.parent._domeTriageShortcutsAttached) { return; }
        window.parent._domeTriageShortcutsAttached = true;

        function findButton(labelStart) {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const matches = buttons.filter(b => b.innerText.trim().startsWith(labelStart));
            // A button with a `help=` tooltip (the numbered reason buttons) renders as TWO DOM
            // nodes -- one real/visible, one a zero-size disconnected leftover -- prefer the
            // visible one instead of trusting DOM order to put the real one first.
            return matches.find(b => b.offsetParent !== null) || matches[0];
        }

        function showFeedback(text) {
            let el = doc.getElementById('_domeTriageKeyToast');
            if (!el) {
                el = doc.createElement('div');
                el.id = '_domeTriageKeyToast';
                el.style.cssText = 'position:fixed; top:10px; left:50%; '
                    + 'transform:translateX(-50%); background:#1a7f4b; color:#fff; '
                    + 'padding:8px 18px; border-radius:6px; font-family:"Source Sans Pro",'
                    + 'sans-serif; font-size:14px; font-weight:600; z-index:999999; '
                    + 'box-shadow:0 2px 12px rgba(0,0,0,0.35); pointer-events:none; opacity:0; '
                    + 'transition:opacity 0.12s ease-out;';
                doc.body.appendChild(el);
            }
            el.textContent = '\\u2328 ' + text;
            el.style.opacity = '1';
            clearTimeout(window.parent._domeTriageToastTimer);
            window.parent._domeTriageToastTimer = setTimeout(function() {
                el.style.opacity = '0';
            }, 900);
        }

        doc.addEventListener('keydown', function(e) {
            const active = doc.activeElement;
            const isTyping = active && (
                active.tagName === 'TEXTAREA' ||
                active.tagName === 'INPUT' ||
                active.isContentEditable
            );
            if (isTyping) { return; }

            const keyMap = Object.assign({
                'p': 'Positive (P)', 'n': 'Negative (N)',
                'u': 'Undeterminable (U)', 's': 'Skip (S)',
            }, """
    + json.dumps(_digit_shortcuts)
    + """);
            const label = keyMap[e.key.toLowerCase()];
            if (!label) { return; }
            const btn = findButton(label);
            if (!btn) {
                showFeedback('"' + e.key.toUpperCase() + '" -- button not found, try again');
                return;
            }
            showFeedback(label);
            btn.click();
            e.preventDefault();
        });
    })();
    </script>
    """,
    height=1,  # st.iframe rejects 0 (StreamlitInvalidHeightError) -- 1px is the smallest valid,
    # effectively-invisible value; unlike the old components.html, 0 isn't accepted here.
)
