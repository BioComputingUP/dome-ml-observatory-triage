"""The accept list for data links found through EBI Search, and the rules that turn an EBI Search
entry into a link.

EBI Search indexes EMBL-EBI's databases and several it mirrors (GEO, dbGaP, the ProteomeXchange
partners, NODE). An entry whose publication field names a paper is a database-side link: the
repository says the entry and the paper belong together, whether or not the paper's text mentions
the accession. Only some of those are assets *from the paper*. Decided 2026-09-14 after the
evaluation recorded in `docs/data_links_sources.md`:

- registries: bio.tools (the paper's software) and the DOME Registry (its ML transparency report);
- deposits: the repositories the paper's data were deposited in, including the domains no positive
  matched yet, so a future batch picks them up;
- supplementary: BioStudies' literature entries (a PMC article's supplementary files), which merge
  with the `S-EPMC` entry `build_data_links.py` already derives.

Every other domain EBI Search returns is rejected at merge time: databases that cite a paper as
curation evidence (UniProt, PDBe-KB, InterPro, GO, EFO, IntAct, Complex Portal, Reactome, Rhea,
ChEBI, ChEMBL, GWAS Catalog, HGNC, G2P, OMIM, Cellosaurus, Ensembl genes, RNAcentral, Rfam, ...),
MeSH, and the derived Expression Atlas experiments and GEO DataSets.

Pure: no I/O, no network. Every URL template was live-checked on 2026-09-14; a resource without a
verified template gets no URL (the record page falls back), never a guessed one.
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple
from urllib.parse import quote

from link_identifiers import url_problems

DEPOSIT = "deposit"
REGISTRY = "registry"
SUPPLEMENTARY = "supplementary"

OBTAINED_BY_XREF = "ebisearch_xref"       # EBI Search's europepmc cross-reference, by PMID
OBTAINED_BY_DOMAIN = "ebisearch_domain"   # a whole-domain dump, matched on PMID / PMCID / DOI


class Domain(NamedTuple):
    slug: str        # the datalinks_resources slug the link is filed under
    kind: str        # deposit | registry | supplementary
    publisher: str   # the repository, as the card names it
    id_scheme: str   # the accession type


DOMAINS: dict[str, Domain] = {
    # registries
    "biotools": Domain("biotools", REGISTRY, "bio.tools", "biotoolsID"),
    "dome-registry": Domain("dome_registry", REGISTRY, "DOME Registry", "DOME Registry ID"),
    # deposits: sequences and projects
    "project": Domain("bioproject", DEPOSIT, "ENA", "BioProject"),
    "earlycause-molecular-sequences": Domain("bioproject", DEPOSIT, "ENA", "BioProject"),
    "sra-study": Domain("ena", DEPOSIT, "ENA", "SRA study"),
    "sra-analysis": Domain("ena", DEPOSIT, "ENA", "SRA analysis"),
    "sra-analysis-covid19": Domain("ena", DEPOSIT, "ENA", "SRA analysis"),
    "sra-analysis-mpox": Domain("ena", DEPOSIT, "ENA", "SRA analysis"),
    "wgs_masters": Domain("ena", DEPOSIT, "ENA", "WGS set"),
    "tsa_masters": Domain("ena", DEPOSIT, "ENA", "TSA set"),
    "tls_masters": Domain("ena", DEPOSIT, "ENA", "TLS set"),
    "emblstandard": Domain("ena", DEPOSIT, "ENA", "ENA sequence"),
    "emblcon": Domain("ena", DEPOSIT, "ENA", "ENA sequence"),
    "embl-covid19": Domain("ena", DEPOSIT, "ENA", "ENA sequence"),
    "embl-pathogen": Domain("ena", DEPOSIT, "ENA", "ENA sequence"),
    "node": Domain("node", DEPOSIT, "NODE", "NODE experiment"),
    # deposits: expression
    "geo": Domain("geo", DEPOSIT, "NCBI GEO", "GEO series"),
    "biostudies-arrayexpress": Domain("arrayexpress", DEPOSIT, "ArrayExpress", "ArrayExpress"),
    "sc-experiments": Domain("expression_atlas", DEPOSIT, "Single Cell Expression Atlas",
                             "Expression Atlas experiment"),
    # deposits: proteomics and metabolomics
    "pride": Domain("pride", DEPOSIT, "PRIDE", "ProteomeXchange"),
    "iprox": Domain("iprox", DEPOSIT, "iProX", "ProteomeXchange"),
    "jpost": Domain("jpost", DEPOSIT, "jPOST", "ProteomeXchange"),
    "panorama": Domain("panorama", DEPOSIT, "Panorama Public", "ProteomeXchange"),
    "massive": Domain("massive", DEPOSIT, "MassIVE", "MassIVE"),
    "gnps": Domain("massive", DEPOSIT, "GNPS", "MassIVE"),
    "metabolights": Domain("metabolights", DEPOSIT, "MetaboLights", "MetaboLights"),
    "metabolights_dataset": Domain("metabolights", DEPOSIT, "MetaboLights", "MetaboLights"),
    # deposits: structures and images
    "pdbe": Domain("pdb", DEPOSIT, "PDBe", "PDB"),
    "emdb": Domain("emdb", DEPOSIT, "EMDB", "EMDB"),
    "empiar": Domain("empiar", DEPOSIT, "EMPIAR", "EMPIAR"),
    "bioimages": Domain("bioimage_archive", DEPOSIT, "BioImage Archive", "BioStudies"),
    # deposits: human genetics and variation
    "ega": Domain("ega", DEPOSIT, "EGA", "EGA"),
    "ega-bycovid": Domain("ega", DEPOSIT, "EGA", "EGA"),
    "dbgap": Domain("dbgap", DEPOSIT, "dbGaP", "dbGaP study"),
    "eva_studies": Domain("eva", DEPOSIT, "EVA", "BioProject"),
    "dgva": Domain("dgva", DEPOSIT, "DGVa", "DGVa"),
    # deposits: models and other studies
    "biomodels": Domain("biomodels", DEPOSIT, "BioModels", "BioModels"),
    "fairdomhub": Domain("fairdomhub", DEPOSIT, "FAIRDOMHub", "FAIRDOMHub"),
    "physiome": Domain("physiome", DEPOSIT, "Physiome Model Repository", "PMR workspace"),
    "cellcollective": Domain("cellcollective", DEPOSIT, "Cell Collective", "Cell Collective"),
    "biostudies-other": Domain("biostudies", DEPOSIT, "BioStudies", "BioStudies"),
    # supplementary
    "biostudies-literature": Domain("biostudies", SUPPLEMENTARY, "BioStudies", "BioStudies"),
}

# Accepted domains over `fetch_ebisearch_domains.py`'s 100,000-entry dump cap, reached per PMID
# through `fetch_ebisearch_xrefs.py detail`. The rest are dumped whole and matched locally.
XREF_DOMAINS = frozenset({
    "project", "sra-study", "sra-analysis", "sra-analysis-covid19", "wgs_masters", "emblstandard",
    "emblcon", "embl-covid19", "embl-pathogen", "geo", "pdbe", "dgva", "biostudies-literature",
})
DUMP_DOMAINS = frozenset(DOMAINS) - XREF_DOMAINS


class IdRule(NamedTuple):
    pattern: re.Pattern
    slug: str
    rewrite: Callable[[re.Match], str]


# An accession's home resource, whichever route or domain produced it. Applied to every link before
# the dedupe so two routes naming one dataset make one link: ArrayExpress imports GEO series as
# `E-GEOD-n`, EGA lists dbGaP studies with a version suffix, BioStudies-Other lists BioModels and
# EMPIAR entries, GNPS lists MassIVE datasets, MetaboLights is indexed twice.
ID_RULES: tuple[IdRule, ...] = (
    IdRule(re.compile(r"^E-GEOD-(\d+)$", re.I), "geo", lambda m: f"GSE{m.group(1)}"),
    IdRule(re.compile(r"^(phs\d{6})(?:\.v\d+\.p\d+)?$", re.I), "dbgap", lambda m: m.group(1).lower()),
    IdRule(re.compile(r"^((?:BIOMD|MODEL)\d{10})$", re.I), "biomodels", lambda m: m.group(1).upper()),
    IdRule(re.compile(r"^(EMPIAR-\d+)$", re.I), "empiar", lambda m: m.group(1).upper()),
    IdRule(re.compile(r"^(S-BIAD\d+)$", re.I), "bioimage_archive", lambda m: m.group(1).upper()),
    IdRule(re.compile(r"^(S-EPMC\d+)$", re.I), "biostudies", lambda m: m.group(1).upper()),
    IdRule(re.compile(r"^(MSV\d{9})$", re.I), "massive", lambda m: m.group(1).upper()),
    IdRule(re.compile(r"^(MTBL[SC]\d+)$", re.I), "metabolights", lambda m: m.group(1).upper()),
)

# A ProteomeXchange dataset is one accession whichever partner hosts it; EBI Search says which.
PXD_RE = re.compile(r"^PXD\d{6}$", re.I)
PXD_HOSTS = ("pride", "iprox", "jpost", "panorama")      # precedence when two claim one id
_PXD_SLUGS = frozenset({"pride", "proteomexchange", *PXD_HOSTS})

_ENA_VIEW = "https://www.ebi.ac.uk/ena/browser/view/{id}"
_BIOSTUDIES = "https://www.ebi.ac.uk/biostudies/studies/{id}"
_PROTEOMEXCHANGE = "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID={id}"
URL_TEMPLATES: dict[str, str] = {
    "biotools": "https://bio.tools/{id}",
    "dome_registry": "https://registry.dome-ml.org/review/{id}",
    "bioproject": _ENA_VIEW,
    "ena": _ENA_VIEW,
    "eva": _ENA_VIEW,                     # an EVA study is an ENA project (PRJEB...)
    "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={id}",
    "arrayexpress": "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{id}",
    "expression_atlas": "https://www.ebi.ac.uk/gxa/sc/experiments/{id}",
    "pride": "https://www.ebi.ac.uk/pride/archive/projects/{id}",
    "iprox": _PROTEOMEXCHANGE,
    "jpost": _PROTEOMEXCHANGE,
    "panorama": _PROTEOMEXCHANGE,
    "massive": "https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession={id}",
    "metabolights": "https://www.ebi.ac.uk/metabolights/{id}",
    "pdb": "https://www.ebi.ac.uk/pdbe/entry/pdb/{id}",
    "emdb": "https://www.ebi.ac.uk/emdb/{id}",
    "empiar": "https://www.ebi.ac.uk/empiar/{id}",
    "bioimage_archive": "https://www.ebi.ac.uk/biostudies/bioimages/studies/{id}",
    "biostudies": _BIOSTUDIES,
    "biomodels": "https://www.biomodels.org/{id}",
    "dbgap": "https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id={id}",
    "node": "https://www.biosino.org/node/experiment/detail/{id}",
}
# No stable per-id page: DGVa variant ids have none; FAIRDOMHub, Physiome and Cell Collective ids
# are not the path of their page, so those links take the entry's own `full_dataset_link`.
NO_URL_TEMPLATE = frozenset({"dgva", "fairdomhub", "physiome", "cellcollective"})

# One page at the source listing every entry of a resource that names the paper. NCBI's Entrez link
# page does (checked 2026-09-14). The EBI Search web UI renders client-side, which a server check
# cannot confirm, so EBI-hosted resources carry none until it has been checked in a browser.
BROWSE_TEMPLATES: dict[str, str] = {
    "geo": "https://www.ncbi.nlm.nih.gov/gds?LinkName=pubmed_gds&from_uid={pmid}",
}

RELATIONSHIPS = {"dome_registry": "IsReviewedBy", "biotools": "IsDescribedBy"}


def canonicalise(slug: str, link_id: str) -> tuple[str, str]:
    """(home resource, canonical id) for an accession; unchanged when no rule knows it."""
    text = (link_id or "").strip()
    for rule in ID_RULES:
        match = rule.pattern.match(text)
        if match:
            return rule.slug, rule.rewrite(match)
    return slug, text


def entry_url(slug: str, link_id: str) -> str | None:
    """The entry's page at its repository, or None when there is no verified template."""
    if slug == "ega":
        kind = "datasets" if link_id.upper().startswith("EGAD") else "studies"
        url = f"https://ega-archive.org/{kind}/{quote(link_id, safe='')}"
    else:
        template = URL_TEMPLATES.get(slug)
        if not template:
            return None
        url = template.format(id=quote(link_id, safe=":-._~"))
    return None if url_problems(url) else url


def browse_url(slug: str, pmid: str | None) -> str | None:
    template = BROWSE_TEMPLATES.get(slug)
    pmid = (pmid or "").strip()
    if not template or not pmid.isdigit():
        return None
    return template.format(pmid=pmid)


def relationship_for(domain: Domain) -> str:
    """DataCite relationType, the vocabulary the Scholix links already use."""
    return RELATIONSHIPS.get(domain.slug, "IsSupplementedBy")


def canonicalise_link(link: dict) -> dict:
    """A link filed under its home resource. A link a rule moves takes its new resource's URL;
    one no rule touches is returned as the same object."""
    slug, link_id = canonicalise(link["resource"], link["id"])
    if (slug, link_id) == (link["resource"], link["id"]):
        return link
    return dict(link, resource=slug, id=link_id, url=entry_url(slug, link_id))


def resolve_pxd_hosts(links: list) -> list:
    """Files every PXD link under the partner EBI Search says hosts it, so a PXD text-mined as PRIDE
    and the same PXD listed by iProX make one iProX link. A PXD no EBI Search entry claims keeps
    its resource. Non-link items (None, the build's PENDING marker) pass through."""
    hosts: dict[str, str] = {}
    for link in links:
        if (isinstance(link, dict) and link.get("source_domain") and link["resource"] in PXD_HOSTS
                and PXD_RE.match(link["id"] or "")):
            key = link["id"].upper()
            current = hosts.get(key)
            if current is None or PXD_HOSTS.index(link["resource"]) < PXD_HOSTS.index(current):
                hosts[key] = link["resource"]
    if not hosts:
        return links
    out = []
    for link in links:
        if isinstance(link, dict) and link["resource"] in _PXD_SLUGS and PXD_RE.match(link["id"] or ""):
            host = hosts.get(link["id"].upper())
            if host and host != link["resource"]:
                link = dict(link, resource=host, url=entry_url(host, link["id"].upper()))
        out.append(link)
    return out
