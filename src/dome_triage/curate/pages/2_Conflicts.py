"""Conflict resolution queue: shows every contributing source's label side-by-side for records
where sources disagree, and lets a curator record the resolved label. Never overwrites
conflicts_for_review.csv in place -- resolutions append to conflict_resolutions.csv instead, so
the original provenance record is preserved (see AGENTS.md).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dome_triage.config import resolve_path
from dome_triage.curate.state import backup_file
from dome_triage.curate.streamlit_helpers import get_config

st.set_page_config(page_title="Conflicts", layout="wide")
st.title("Conflicts")

cfg = get_config()
conflicts_path = cfg.path("conflicts_for_review")
resolutions_path = resolve_path(cfg.pipeline["curation"]["conflict_resolutions"])

if not conflicts_path.exists():
    st.info("No conflicts_for_review.csv yet -- run `dome-triage dedupe consolidate` first.")
    st.stop()

conflicts = pd.read_csv(conflicts_path, dtype=str)
resolved_ids: set[str] = set()
if resolutions_path.exists():
    resolved_ids = set(pd.read_csv(resolutions_path, dtype=str)["record_id"])

pending = conflicts[~conflicts["record_id"].isin(resolved_ids)]
record_ids = pending["record_id"].unique().tolist()

st.caption(f"{len(record_ids)} conflicting record(s) remaining")

if not record_ids:
    st.success("No conflicts remaining.")
    st.stop()

record_id = record_ids[0]
rows = pending[pending["record_id"] == record_id]

st.subheader(rows.iloc[0].get("title") or "(no title)")
with st.container(height=250):
    st.markdown(rows.iloc[0].get("abstract") or "*(no abstract)*", unsafe_allow_html=True)

st.table(rows[["source_name", "source_label", "source_label_confidence", "matched_on"]])

with st.form("conflict_form", clear_on_submit=True):
    resolution = st.radio(
        "Resolved label", ["positive", "negative", "still_uncertain"], horizontal=True
    )
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Submit resolution")

if submitted:
    row = {
        "record_id": record_id,
        "resolution": resolution,
        "notes": notes,
        "curator": st.session_state.get("curator_name", "unknown"),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    backup_file(resolutions_path)
    resolutions_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not resolutions_path.exists()
    pd.DataFrame([row]).to_csv(resolutions_path, mode="a", header=header_needed, index=False)
    st.rerun()
