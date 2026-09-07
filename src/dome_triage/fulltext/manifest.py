"""Full-text manifest: scans the sibling repos' PDF directories (`fulltext_roots` in
configs/sources.yaml) to record which PMCIDs have full text available and where, WITHOUT ever
copying PDFs into this repo -- see AGENTS.md. `fetch_fulltext_xml` provides an independent
Europe-PMC-based fallback for reproducing full text on a machine without those sibling repos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from dome_triage.ingest.id_mapping import clean_pmcid

EPMC_FULLTEXT_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def _flat_main_pdf_entries(root: Path) -> list[tuple[str, Path]]:
    return [(p.name.split("_main.pdf")[0], p) for p in root.glob("PMC*_main.pdf")]


def _pmcid_subdir_entries(root: Path) -> list[tuple[str, Path]]:
    entries = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("PMC"):
            main_pdf = sub / f"{sub.name}_main.pdf"
            if main_pdf.exists():
                entries.append((sub.name, main_pdf))
    return entries


_LAYOUTS: dict[str, Callable[[Path], list[tuple[str, Path]]]] = {
    "flat_main_pdf": _flat_main_pdf_entries,
    "pmcid_subdir": _pmcid_subdir_entries,
}


def build_manifest(sources_cfg: dict) -> pd.DataFrame:
    rows = []
    for root_cfg in sources_cfg.get("fulltext_roots", []):
        root = Path(root_cfg["path"])
        if not root.exists():
            continue
        entries = _LAYOUTS[root_cfg["pdf_layout"]](root)
        for pmcid, pdf_path in entries:
            rows.append(
                {
                    "pmcid": clean_pmcid(pmcid),
                    "absolute_path": str(pdf_path),
                    "source_root": root_cfg["name"],
                    "size_bytes": pdf_path.stat().st_size,
                    "exists_ok": pdf_path.exists(),
                }
            )
    columns = ["pmcid", "absolute_path", "source_root", "size_bytes", "exists_ok"]
    return pd.DataFrame(rows, columns=columns)


def fetch_fulltext_xml(pmcid: str, timeout: int = 30) -> Optional[str]:
    """Reproducibility fallback: fetches full-text XML directly from Europe PMC (works for
    open-access content) independent of the sibling repos' local PDF directories. Ported from the
    proven pattern in MLit-Triage-Nextflow/Download_DOME_Registry_Contents.ipynb."""
    normalized = clean_pmcid(pmcid)
    if not normalized:
        return None
    response = requests.get(EPMC_FULLTEXT_XML_URL.format(pmcid=normalized), timeout=timeout)
    if response.status_code != 200:
        return None
    return response.text


def save_fulltext_xml(pmcid: str, output_dir: Path) -> Optional[Path]:
    xml = fetch_fulltext_xml(pmcid)
    if xml is None:
        return None
    normalized = clean_pmcid(pmcid)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized}.xml"
    output_path.write_text(xml, encoding="utf-8")
    return output_path
