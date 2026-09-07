"""ID normalization + DOI<->PMCID mapping.

The cleaning rules here exist because of real dirt confirmed in the source data: PMIDs pandas-cast
to float and stored as e.g. "35582656.0" (positive_entries.csv), DOIs stored with a
"https://doi.org/" prefix or "doi:" prefix and inconsistent casing, and PMCIDs stored without the
"PMC" prefix in some sources. Every loader in source_loaders.py must pass IDs through these
functions before they reach dedupe/keys.py -- clustering correctness depends on it.
"""

from __future__ import annotations

import re
from typing import Optional

import requests

_PMID_RE = re.compile(r"\d+")
_DOI_PREFIX_RE = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)
_DOI_SCHEME_RE = re.compile(r"^doi:\s*", re.IGNORECASE)

NCBI_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def clean_pmid(value: object) -> Optional[str]:
    """Normalize a PMID, stripping pandas float-cast suffixes (e.g. "35582656.0" -> "35582656")."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "-", ""):
        return None
    match = _PMID_RE.match(text)
    return match.group(0) if match else None


def clean_doi(value: object) -> Optional[str]:
    """Normalize a DOI: strip URL/scheme prefixes, lowercase, strip whitespace."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "-", ""):
        return None
    text = _DOI_PREFIX_RE.sub("", text)
    text = _DOI_SCHEME_RE.sub("", text)
    text = text.strip().strip("/").lower()
    return text or None


def clean_pmcid(value: object) -> Optional[str]:
    """Normalize a PMCID: uppercase, ensure a PMC prefix, strip whitespace."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in ("NAN", "NONE", "-", ""):
        return None
    if not text.startswith("PMC"):
        if text.isdigit():
            text = f"PMC{text}"
        else:
            return None
    return text


def doi_to_pmcid(doi: str, tool: str = "dome-triage", email: str = "") -> Optional[str]:
    """Resolve a DOI to a PMCID via the NCBI ID Converter API. Returns None on any failure or
    no-match -- callers should route unresolved records to unresolved_needs_id_lookup.csv rather
    than treating this as fatal."""
    params = {"tool": tool, "email": email, "ids": doi, "format": "json"}
    try:
        resp = requests.get(NCBI_IDCONV_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    records = data.get("records", [])
    if not records or "pmcid" not in records[0]:
        return None
    return clean_pmcid(records[0]["pmcid"])
