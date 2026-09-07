"""MeSH-first ontology tagging (ROADMAP Phase 2): MeSH headings are already present on many
Europe PMC records and need no inference -- just extraction. EDAM/domain-science mapping is a
later, optional enhancement layered on top of what's captured here, not the starting point.

Confirmed via a live `resultType=core` fetch: `meshHeadingList.meshHeading` is a list of
`{descriptorName, majorTopic_YN, meshQualifierList?}` dicts, present on MEDLINE-sourced records
but NOT guaranteed even there (MeSH indexing lags publication) -- always treat as optional.
"""

from __future__ import annotations


def extract_mesh_headings(epmc_core_result: dict) -> list[str]:
    """Pulls MeSH descriptor names from a single Europe PMC `resultType=core` search result.
    Returns an empty list if the record has no MeSH headings (common for very recent articles
    or non-MEDLINE sources) -- never raises on absence."""
    headings = (epmc_core_result.get("meshHeadingList") or {}).get("meshHeading") or []
    return [h["descriptorName"] for h in headings if h.get("descriptorName")]
