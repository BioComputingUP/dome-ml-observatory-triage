"""Streamlit curation app entrypoint. Run via `docker compose up curate` (see README.md) or
`streamlit run src/dome_triage/curate/app.py` locally. The actual work happens in curate/pages/ --
Streamlit auto-discovers files there as sidebar navigation (Curate / Conflicts / Keyword Review).
"""

from __future__ import annotations

import streamlit as st

from dome_triage.curate.streamlit_helpers import get_config, get_session

st.set_page_config(page_title="dome-triage curation", layout="wide")

st.title("dome-triage — human curation")

st.session_state.setdefault("curator_name", "")
st.session_state["curator_name"] = st.text_input(
    "Curator name", value=st.session_state["curator_name"]
)

if not st.session_state["curator_name"]:
    st.warning("Enter your curator name above to begin.")
    st.stop()

cfg = get_config()
if not cfg.path("canonical_dataset").exists():
    st.info(
        "No canonical_dataset.csv yet -- run the ingest/dedupe pipeline first "
        "(see README.md Quickstart)."
    )
    st.stop()

session = get_session()
stats = session.stats()
col1, col2, col3 = st.columns(3)
col1.metric("Total records", stats["total"])
col2.metric("Decided", stats["decided"])
col3.metric("Remaining", stats["remaining"])

st.info("Use the sidebar to open **Curate**, **Conflicts**, or **Keyword Review**.")
