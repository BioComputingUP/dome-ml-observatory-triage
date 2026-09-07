"""Step 19c (standard-curation-route build): curates the 192 real, individually-verified
BERT/foundation-model/agentic-AI/biomedical-LLM candidate papers found for the seed file
(`llm_language_model_seed_pmids.csv` -- see STEPS_Progress.md's "Exact search provenance" section
for exactly how they were found). Supersedes the earlier `pmid,name,url,use` mark-and-bulk-merge
design -- per explicit instruction, these candidates get the same one-at-a-time P/N/U/S curation
every other batch in this dataset gets (title, journal, year, MeSH terms, full abstract), not a
pre-judged CSV column.

Deliberately simple compared to `4_Original_Cohort_Review.py`: no sampling, no label filter, no
review-term down-weighting toggle -- this pool is already small (~192 rows, one-time fetched by
`ingest fetch-llm-seed-pool`), so the point is just to work through all of it, same shape as
`1_Curate.py`'s core per-record display but without that page's negative-reason picker/keyboard-
shortcut script (kept out to match `4_Original_Cohort_Review.py`'s simpler precedent for a
dedicated secondary-batch review page, not the main queue's).

Decisions write to `data/processed/llm_seed_review_events.csv` -- a separate event log, same
reasoning as every other dedicated review page in this app. `curate materialize-llm-seed-review`
folds it into `canonical_dataset.csv`, reusing `materialize_events()` with `bulk_pool_path` pointed
at the fetched candidate pool -- every decided record gets inserted as a genuinely new row with
full provenance (title/abstract/journal/year/MeSH/pub types/source_name="manual_llm_language_model
_seed") via the exact same mechanism the main Curate page's "Full AI/ML bulk pool" browsing mode
already uses for records not yet in canonical_dataset.csv, not a bespoke merge path.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dome_triage.config import resolve_path
from dome_triage.curate.streamlit_helpers import get_config, get_llm_seed_review_session

st.set_page_config(page_title="LLM Seed Review", layout="wide")
st.title("LLM Seed Review")
st.caption(
    "Step 19c: curate the 192 real, individually-verified BERT/foundation-model/agentic-AI/"
    "biomedical-LLM candidate papers found for the seed file -- same P/N/U/S workflow as every "
    "other batch in this dataset."
)

if not st.session_state.get("curator_name"):
    st.warning("Set your curator name on the Home page first.")
    st.stop()

cfg = get_config()
pool_path = cfg.path("llm_seed_candidate_pool")
if not pool_path.exists():
    st.info(
        "No fetched candidate pool yet -- run `docker compose run --rm pipeline dome-triage "
        "ingest fetch-llm-seed-pool` first (fetches full EPMC metadata for every PMID in "
        "data/processed/llm_language_model_seed_pmids.csv)."
    )
    st.stop()

events_path = resolve_path(cfg.pipeline["curation"]["llm_seed_review_events"])
decided_ids: set[str] = set()
if events_path.exists():
    decided_ids = set(pd.read_csv(events_path, usecols=["record_id"], dtype=str)["record_id"])

session = get_llm_seed_review_session()
st.caption(
    f"Pool: **{session.total() + len(decided_ids):,}** fetched candidates -- "
    f"**{len(decided_ids):,}** already decided (all-time, across every session)."
)

nav_prog, nav_back, nav_fwd = st.columns([6, 1, 1])
nav_prog.caption(f"{session.remaining()} of {session.total()} remaining")
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
    st.success("Every fetched candidate has been decided. Run `curate materialize-llm-seed-review` "
               "to fold these decisions into canonical_dataset.csv.")
    st.stop()

prior_decision = session.current_record_prior_decision()
if prior_decision:
    st.info(f"You already marked this record **{prior_decision}** this session. Reviewing again.")

# ---------------------------------------------------------------------------------------------
# Paper display -- same fields/layout as 1_Curate.py's core display (title, journal, year, MeSH,
# full abstract, no fixed-height scroll box).
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

mesh_raw = record.get("mesh_headings")
if isinstance(mesh_raw, str) and mesh_raw.strip() not in ("", "[]"):
    try:
        mesh_list = json.loads(mesh_raw)
    except json.JSONDecodeError:
        mesh_list = []
    if mesh_list:
        st.markdown(f"**MeSH terms:** {', '.join(mesh_list)}")

st.markdown(record.get("abstract") or "*(no abstract available)*", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------
# Decision -- same immediate-submit P/N/U/S pattern as every other page in this app.
# ---------------------------------------------------------------------------------------------


def _submit(decision: str):
    def _callback():
        notes = st.session_state.get("llm_seed_notes", "")
        session.record_decision(decision, notes=notes)
        st.session_state["llm_seed_notes"] = ""

    return _callback


btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
btn_col1.button("Positive (P)", on_click=_submit("positive"), use_container_width=True, type="primary")
btn_col2.button("Negative (N)", on_click=_submit("negative"), use_container_width=True)
btn_col3.button("Undeterminable (U)", on_click=_submit("undeterminable"), use_container_width=True)
btn_col4.button("Skip (S)", on_click=_submit("skipped"), use_container_width=True)

st.text_area("Notes (optional)", key="llm_seed_notes")
