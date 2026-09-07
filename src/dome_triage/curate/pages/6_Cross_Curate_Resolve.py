"""Step 20: reviews records where DeepSeek's independent, blind classification disagreed with the
existing human-curated label -- the final human tie-break this second-curator pass exists to
produce. Shows the human's current label side by side with every disagreeing tier's classification
and rationale (mirrors `2_Conflicts.py`'s side-by-side table), plus the full paper text (mirrors
`4_Original_Cohort_Review.py`/`5_LLM_Seed_Review.py`'s display + P/N/U/S submit pattern). Decisions
write to their own event log (`cross_curate_resolution_events.csv`) via
`CurationSession.record_decision()` -- the same backup-before-write, tested mechanism every other
dedicated review page in this app already uses, never a raw `st.form` + manual CSV append.

`curate materialize-cross-curate-resolutions` folds this event log into `canonical_dataset.csv`,
reusing `materialize_events()` unchanged -- every record here already exists in
canonical_dataset.csv by construction (it's filtered FROM that file), so no `bulk_pool_path`
fallback is needed, same as `4_Original_Cohort_Review.py`.

Page numbered **6**, not 5 -- `5_LLM_Seed_Review.py` (Step 19c) already claims 5.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dome_triage.config import resolve_path
from dome_triage.curate.cohort_filters import build_disagreement_queue
from dome_triage.curate.streamlit_helpers import get_config, get_cross_curate_resolve_session

st.set_page_config(page_title="Cross Curate Resolve", layout="wide")
st.title("Cross Curate Resolve")
st.caption(
    "Step 20: records where DeepSeek's independent, blind second-curator pass disagreed with the "
    "existing human-curated label. DeepSeek never saw your decision, MeSH terms, or notes -- only "
    "title/abstract/journal/year -- and had no web/search access on its primary pass. Review each "
    "disagreement below and make the final call."
)

if not st.session_state.get("curator_name"):
    st.warning("Set your curator name on the Home page first.")
    st.stop()

cfg = get_config()
canonical_path = cfg.path("canonical_dataset")
events_path = cfg.path("llm_classification_events")

if not events_path.exists():
    st.info(
        "No DeepSeek classification events yet -- run Step 20's `llm-classify classify` first "
        "(see STEPS_Progress.md)."
    )
    st.stop()

dataset = pd.read_csv(canonical_path, dtype=str)
llm_events = pd.read_csv(events_path, dtype=str)

tier_choice = st.selectbox("Tier", ["Either tier disagrees", "flash", "pro"], index=0)
tier = None if tier_choice == "Either tier disagrees" else tier_choice

resolve_events_path = resolve_path(cfg.pipeline["curation"]["cross_curate_resolution_events"])
resolved_ids: set[str] = set()
if resolve_events_path.exists():
    resolved_ids = set(pd.read_csv(resolve_events_path, usecols=["record_id"], dtype=str)["record_id"])

# Already-resolved records are excluded from the active queue -- otherwise a record whose final
# decision UPHELD the original label (human explicitly agreed with the human, not DeepSeek) would
# resurface here forever, since label != llm_classification for it by construction even after
# resolution. See `build_disagreement_queue`'s docstring for the full reasoning.
queue = build_disagreement_queue(dataset, llm_events, tier=tier, exclude_ids=resolved_ids)

if queue.empty:
    st.success("No unresolved disagreements for this tier selection.")
    st.stop()

view_df = dataset[dataset["record_id"].isin(queue["record_id"].unique())].drop_duplicates(subset="record_id")

session = get_cross_curate_resolve_session(view_df)
st.caption(
    f"Disagreement queue: **{session.total():,}** unresolved records -- "
    f"**{len(resolved_ids):,}** already resolved (all-time, excluded from this queue)."
)

nav_prog, nav_back, nav_fwd = st.columns([6, 1, 1])
nav_prog.caption(f"{session.remaining()} of {session.total()} remaining")
nav_back.button(
    "< Back", disabled=not session.can_go_back(), on_click=session.go_back, use_container_width=True
)
nav_fwd.button(
    "Forward >", disabled=not session.can_go_forward(), on_click=session.go_forward, use_container_width=True
)

record = session.current_record()
if record is None:
    st.success(
        "Every disagreement has been resolved. Run `curate materialize-cross-curate-resolutions` "
        "to fold these decisions into canonical_dataset.csv."
    )
    st.stop()

prior_decision = session.current_record_prior_decision()
if prior_decision:
    st.info(f"You already marked this record **{prior_decision}** this session. Reviewing again.")

# ---------------------------------------------------------------------------------------------
# Paper display -- same fields/layout as 4_Original_Cohort_Review.py/5_LLM_Seed_Review.py.
# ---------------------------------------------------------------------------------------------
st.markdown(f"### {record.get('title') or '(no title)'}", unsafe_allow_html=True)


def _display_year(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "(unknown)"


meta_col1, meta_col2 = st.columns(2)
meta_col1.markdown(f"**Journal:** {record.get('journal') or '(unknown)'}", unsafe_allow_html=True)
meta_col2.markdown(f"**Year:** {_display_year(record.get('year'))}")

st.markdown(record.get("abstract") or "*(no abstract available)*", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------
# Side-by-side human-vs-DeepSeek table -- mirrors 2_Conflicts.py's rows[["source_name", ...]]
# display. record_disagreements has one row per tier that disagreed with the human label.
# ---------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("**Human label vs. DeepSeek classification:**")
record_disagreements = queue[queue["record_id"] == record["record_id"]]
display_rows = [{"Source": "Human (current label)", "Classification": record.get("label"), "Rationale": ""}]
display_rows += [
    {
        "Source": f"DeepSeek ({row['llm_tier']})",
        "Classification": row["llm_classification"],
        "Rationale": row["llm_rationale"],
    }
    for _, row in record_disagreements.iterrows()
]
st.table(pd.DataFrame(display_rows))

# ---------------------------------------------------------------------------------------------
# Decision -- same immediate-submit P/N/U/S pattern as every other page in this app.
# ---------------------------------------------------------------------------------------------


def _submit(decision: str):
    def _callback():
        notes = st.session_state.get("cross_curate_resolve_notes", "")
        session.record_decision(decision, notes=notes)
        st.session_state["cross_curate_resolve_notes"] = ""

    return _callback


btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
btn_col1.button("Positive (P)", on_click=_submit("positive"), use_container_width=True, type="primary")
btn_col2.button("Negative (N)", on_click=_submit("negative"), use_container_width=True)
btn_col3.button("Undeterminable (U)", on_click=_submit("undeterminable"), use_container_width=True)
btn_col4.button("Skip (S)", on_click=_submit("skipped"), use_container_width=True)

st.text_area("Notes (optional)", key="cross_curate_resolve_notes")
