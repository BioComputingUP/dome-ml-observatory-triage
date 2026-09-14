"""The finite set of resources a Europe PMC data link can point at, and how a raw link is mapped
onto one of them.

Europe PMC names the same resource three different ways depending on the route: the search
record's `tmAccessionTypeList` says `pdb`, the annotations API says `subType: "PDBe"` (or nothing,
for an accession mined from a supplementary file by BioStudies -- then the identifiers.org URI
says `pdbe/pdb`), and the `/datalinks` Scholix payload says `IDScheme: "PDB"` with a `Publisher`.
A text-mined DOI is a resource only by its prefix: `10.5061` is Dryad, `10.5281` Zenodo, and a
DOI in the reference list is a citation, not data. `slug()` reduces all of that to one stable key
per resource, which is what the Observatory keys its cards, icons and facet on.

Every entry: slug -> (label, category). Categories follow the groupings Europe PMC's own Data tab
uses, so the record page can group cards the way a reader has already seen elsewhere. The table is
completed from what the real corpus returns (`build_data_links.py --report-only` tabulates every
unmapped scheme, publisher and DOI prefix); an unknown scheme still yields a slug, labelled with
the raw name and filed under "Other", so nothing is dropped for being new.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class Resource(NamedTuple):
    slug: str
    label: str
    category: str


NUCLEOTIDE = "Nucleotide Sequences"
PROTEIN_SEQ = "Protein Sequences"
STRUCTURES = "Protein Structures"
EXPRESSION = "Gene Expression"
VARIATION = "Genomic Variation"
GENOMES = "Genomes & Assemblies"
TRIALS = "Clinical Trials"
CHEMICALS = "Chemicals & Compounds"
PROTEOMICS = "Proteomics"
METABOLOMICS = "Metabolomics"
IMAGING = "Imaging"
INTERACTIONS = "Pathways & Interactions"
ANNOTATIONS = "Ontologies & Annotations"
CELL_LINES = "Cell Lines & Reagents"
CITATIONS = "Data Citations"
SUPPLEMENTARY = "Supplementary Material"
CODE = "Code & Notebooks"
MODELS = "Models"
OTHER = "Other"

RESOURCES: dict[str, Resource] = {r.slug: r for r in [
    Resource("ena", "European Nucleotide Archive", NUCLEOTIDE),
    Resource("ena_assembly", "ENA genome assembly", GENOMES),
    Resource("refseq", "NCBI RefSeq", NUCLEOTIDE),
    Resource("bioproject", "BioProject", GENOMES),
    Resource("biosample", "BioSample", GENOMES),
    Resource("ensembl", "Ensembl", GENOMES),
    Resource("igsr", "IGSR / 1000 Genomes", GENOMES),
    Resource("gisaid", "GISAID", NUCLEOTIDE),
    Resource("mgnify", "MGnify", GENOMES),
    Resource("uniprot", "UniProt", PROTEIN_SEQ),
    Resource("uniparc", "UniParc", PROTEIN_SEQ),
    Resource("pfam", "Pfam", PROTEIN_SEQ),
    Resource("interpro", "InterPro", PROTEIN_SEQ),
    Resource("rfam", "Rfam", NUCLEOTIDE),
    Resource("rnacentral", "RNAcentral", NUCLEOTIDE),
    Resource("treefam", "TreeFam", PROTEIN_SEQ),
    Resource("pdb", "Protein Data Bank in Europe", STRUCTURES),
    Resource("emdb", "Electron Microscopy Data Bank", STRUCTURES),
    Resource("empiar", "EMPIAR", IMAGING),
    Resource("alphafold", "AlphaFold DB", STRUCTURES),
    Resource("cath", "CATH", STRUCTURES),
    Resource("geo", "Gene Expression Omnibus", EXPRESSION),
    Resource("arrayexpress", "ArrayExpress / BioStudies", EXPRESSION),
    Resource("hpa", "Human Protein Atlas", EXPRESSION),
    Resource("refsnp", "dbSNP", VARIATION),
    Resource("dbgap", "dbGaP", VARIATION),
    Resource("ega", "European Genome-phenome Archive", VARIATION),
    Resource("gwas", "GWAS Catalog", VARIATION),
    Resource("omim", "OMIM", VARIATION),
    Resource("orphanet", "Orphanet", ANNOTATIONS),
    Resource("hgnc", "HGNC", ANNOTATIONS),
    Resource("clinicaltrials", "ClinicalTrials.gov", TRIALS),
    Resource("eudract", "EU Clinical Trials Register", TRIALS),
    Resource("chembl", "ChEMBL", CHEMICALS),
    Resource("chebi", "ChEBI", CHEMICALS),
    Resource("rhea", "Rhea", CHEMICALS),
    Resource("brenda", "BRENDA", CHEMICALS),
    Resource("pride", "PRIDE", PROTEOMICS),
    Resource("metabolights", "MetaboLights", METABOLOMICS),
    Resource("bioimage_archive", "BioImage Archive", IMAGING),
    Resource("reactome", "Reactome", INTERACTIONS),
    Resource("intact", "IntAct", INTERACTIONS),
    Resource("mint", "MINT", INTERACTIONS),
    Resource("complexportal", "Complex Portal", INTERACTIONS),
    Resource("biomodels", "BioModels", MODELS),
    Resource("go", "Gene Ontology", ANNOTATIONS),
    Resource("efo", "Experimental Factor Ontology", ANNOTATIONS),
    Resource("cellosaurus", "Cellosaurus", CELL_LINES),
    Resource("rrid", "Research Resource Identifiers", CELL_LINES),
    Resource("ebisc", "EBiSC", CELL_LINES),
    Resource("hipsci", "HipSci", CELL_LINES),
    Resource("hpscreg", "hPSCreg", CELL_LINES),
    Resource("coriell", "Coriell Biorepository", CELL_LINES),
    Resource("proteomexchange", "ProteomeXchange", PROTEOMICS),
    Resource("biostudies", "BioStudies", SUPPLEMENTARY),
    Resource("zenodo", "Zenodo", CITATIONS),
    Resource("dryad", "Dryad", CITATIONS),
    Resource("figshare", "figshare", CITATIONS),
    Resource("osf", "Open Science Framework", CITATIONS),
    Resource("mendeley_data", "Mendeley Data", CITATIONS),
    Resource("dataverse", "Dataverse", CITATIONS),
    Resource("pangaea", "PANGAEA", CITATIONS),
    Resource("gigadb", "GigaDB", CITATIONS),
    Resource("morphosource", "MorphoSource", CITATIONS),
    Resource("gbif", "GBIF", CITATIONS),
    Resource("tcia", "The Cancer Imaging Archive", IMAGING),
    Resource("icpsr", "ICPSR", CITATIONS),
    Resource("ieee_dataport", "IEEE DataPort", CITATIONS),
    Resource("4tu", "4TU.ResearchData", CITATIONS),
    Resource("edinburgh_datashare", "Edinburgh DataShare", CITATIONS),
    Resource("apollo", "Apollo (Cambridge)", CITATIONS),
    Resource("code_ocean", "Code Ocean", CODE),
    Resource("kaggle", "Kaggle", CITATIONS),
    Resource("huggingface", "Hugging Face", MODELS),
    Resource("github", "GitHub", CODE),
    Resource("software_heritage", "Software Heritage", CODE),
    Resource("doi", "Data DOI", CITATIONS),
]}

# Every spelling seen so far for a scheme / accession type / publisher, normalised by `_norm`.
ALIASES: dict[str, str] = {
    "ena": "ena", "gen": "ena", "genbank": "ena", "embl": "ena", "enaembl": "ena",
    "nucleotide": "ena", "ddbj": "ena", "insdc": "ena",
    "gca": "ena_assembly", "assembly": "ena_assembly",
    "refseq": "refseq", "bioproject": "bioproject", "biosample": "biosample",
    "ensembl": "ensembl", "igsr": "igsr", "gisaid": "gisaid", "metagenomics": "mgnify",
    "mgnify": "mgnify",
    "uniprot": "uniprot", "uniprotkb": "uniprot", "swissprot": "uniprot", "uniparc": "uniparc",
    "pfam": "pfam", "interpro": "interpro", "rfam": "rfam", "rnacentral": "rnacentral",
    "treefam": "treefam",
    "pdb": "pdb", "pdbe": "pdb", "wwpdb": "pdb", "rcsb": "pdb", "emdb": "emdb",
    "empiar": "empiar", "alphafold": "alphafold", "alphafolddb": "alphafold", "cath": "cath",
    "geo": "geo", "arrayexpress": "arrayexpress", "arxpr": "arrayexpress", "hpa": "hpa",
    "refsnp": "refsnp", "dbsnp": "refsnp", "snp": "refsnp", "dbgap": "dbgap", "ega": "ega",
    "gwas": "gwas", "gwascatalog": "gwas", "omim": "omim", "orphadata": "orphanet",
    "orphanet": "orphanet", "hgnc": "hgnc",
    "nct": "clinicaltrials", "clinicaltrials": "clinicaltrials", "clinicaltrialsgov": "clinicaltrials",
    "eudract": "eudract", "euctr": "eudract",
    "chembl": "chembl", "chebi": "chebi", "rhea": "rhea", "brenda": "brenda",
    "pxd": "pride", "pride": "pride", "metabolights": "metabolights", "mtbls": "metabolights",
    "bia": "bioimage_archive", "bioimagearchive": "bioimage_archive",
    "reactome": "reactome", "intact": "intact", "mint": "mint", "complexportal": "complexportal",
    "biomodels": "biomodels", "go": "go", "geneontology": "go", "efo": "efo",
    "cellosaurus": "cellosaurus", "rrid": "rrid", "ebisc": "ebisc", "hipsci": "hipsci",
    "hpscreg": "hpscreg",
    "biostudies": "biostudies", "zenodo": "zenodo", "dryad": "dryad", "figshare": "figshare",
    "osf": "osf", "openscienceframework": "osf", "mendeleydata": "mendeley_data",
    "mendeley": "mendeley_data", "dataverse": "dataverse", "harvarddataverse": "dataverse",
    "pangaea": "pangaea", "gigadb": "gigadb", "gigascience": "gigadb",
    "morphosource": "morphosource", "gbif": "gbif", "tcia": "tcia",
    "thecancerimagingarchive": "tcia", "icpsr": "icpsr", "ieeedataport": "ieee_dataport",
    "4turesearchdata": "4tu", "codeocean": "code_ocean", "kaggle": "kaggle",
    "huggingface": "huggingface", "github": "github", "softwareheritage": "software_heritage",
    "doi": "doi",
    # Spellings measured in the 2026-09-14 corpus build (annotations subType / identifiers.org path).
    "igsr1000genomes": "igsr", "biosamples": "biosample", "coriell": "coriell",
    "px": "proteomexchange", "proteomexchange": "proteomexchange",
    "chemblcompound": "chembl", "chembltarget": "chembl", "insdcgca": "ena_assembly",
    "egastudy": "ega", "egadataset": "ega", "euclinicaltrials": "eudract",
    "ebimetagenomics": "mgnify", "biomodelsdb": "biomodels",
}

# DOI prefix -> resource, for a text-mined or cited DOI. Only data / code repositories belong
# here: a prefix absent from this table is a literature DOI (or an unknown one, which the report
# tabulates so the table can grow) and is not turned into a link.
DOI_PREFIXES: dict[str, str] = {
    "10.5281": "zenodo",
    "10.5061": "dryad",
    "10.6084": "figshare",
    "10.17605": "osf",
    "10.17632": "mendeley_data",
    "10.7910": "dataverse",
    "10.1594": "pangaea",
    "10.5524": "gigadb",
    "10.17602": "morphosource",
    "10.15468": "gbif",
    "10.7937": "tcia",
    "10.3886": "icpsr",
    "10.21227": "ieee_dataport",
    "10.4121": "4tu",
    "10.7488": "edinburgh_datashare",
    "10.17863": "apollo",
    "10.24433": "code_ocean",
    "10.34740": "kaggle",
    "10.57967": "huggingface",
    "10.25504": "doi",       # FAIRsharing records: a data resource description, kept generic
}

# A DOI prefix several resources share, split by the start of the suffix. 10.6019 is EMBL-EBI's:
# `10.6019/PXD024968` is PRIDE, `10.6019/EMPIAR-10164` is EMPIAR (13 corpus links were filed under
# PRIDE before this rule, 2026-09-14). Anything else under the prefix is a generic data DOI.
DOI_SUFFIX_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "10.6019": (("PXD", "pride"), ("EMPIAR", "empiar")),
}

# Annotation sections whose DOIs are citations of literature, never data of this paper.
REFERENCE_SECTIONS = ("references",)

_DOI_PREFIX_RE = re.compile(r"(10\.\d{4,9})/")
_IDENTIFIERS_ORG_RE = re.compile(r"identifiers\.org/(?:ebi/)?([A-Za-z0-9_.-]+?)(?:[:/]|$)")


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def doi_prefix(doi: str | None) -> str | None:
    match = _DOI_PREFIX_RE.search(doi or "")
    return match.group(1) if match else None


# BioStudies' supplementary-file mining links some accessions straight to the resource's own site,
# not through identifiers.org, and with no subType (measured over the first 200,000 corpus records,
# 2026-09-14: gisaid.org/EPI_ISL 33,414, omim.org/entry 9,556, proteinatlas.org 47, ebi.ac.uk/pdbe 13).
# Checked in order, so a more specific path goes before its host.
HOST_SCHEMES: tuple[tuple[str, str], ...] = (
    ("gisaid.org", "gisaid"),
    ("omim.org", "omim"),
    ("proteinatlas.org", "hpa"),
    ("ebi.ac.uk/pdbe/emdb/empiar", "empiar"),   # before ebi.ac.uk/pdbe: EMPIAR pages live under it
    ("ebi.ac.uk/pdbe/emdb", "emdb"),
    ("ebi.ac.uk/emdb", "emdb"),
    ("ebi.ac.uk/pdbe", "pdb"),
    ("rcsb.org", "pdb"),
    ("ebi.ac.uk/ena", "ena"),
    ("ebi.ac.uk/biostudies", "biostudies"),
    ("ncbi.nlm.nih.gov/geo", "geo"),
    ("ncbi.nlm.nih.gov/snp", "refsnp"),
    ("uniprot.org", "uniprot"),
    ("clinicaltrials.gov", "clinicaltrials"),
)


def scheme_from_uri(uri: str | None) -> str | None:
    """`http://identifiers.org/pdbe/pdb:6w6w` -> `pdbe`; `http://identifiers.org/geo:GSE1` -> `geo`;
    `http://identifiers.org/doi:10.1/x` -> `doi`; `http://gisaid.org/EPI_ISL/1` -> `gisaid`. The
    annotations API omits `subType` for an accession mined from a supplementary file, and the URI is
    the only type information left."""
    match = _IDENTIFIERS_ORG_RE.search(uri or "")
    if match:
        return match.group(1)
    lowered = (uri or "").lower()
    for host, scheme in HOST_SCHEMES:
        if host in lowered:
            return scheme
    return None


_PAREN_RE = re.compile(r"^(.*?)\s*\((.*?)\)\s*$")


def _spellings(value: str | None) -> list[str]:
    """A name as given, and for "Gene Ontology (GO)" also "Gene Ontology" and "GO"."""
    if not value:
        return []
    match = _PAREN_RE.match(value)
    return [value, match.group(1), match.group(2)] if match else [value]


def slug(scheme: str | None, publisher: str | None = None, doi: str | None = None) -> str | None:
    """The resource a link belongs to, or None when it is not a data link at all (a literature
    DOI). Tries the scheme, then the publisher (each also without and inside any parenthesis),
    then the DOI prefix, then falls back to a slug of the raw scheme so an unlisted resource is
    kept rather than lost."""
    for candidate in (scheme, publisher):
        for spelling in _spellings(candidate):
            key = _norm(spelling)
            if key and key != "doi" and key in ALIASES:
                return ALIASES[key]
    if _norm(scheme) == "doi" or doi:
        prefix = doi_prefix(doi)
        if prefix in DOI_SUFFIX_RULES:
            suffix = (doi or "").split(prefix + "/", 1)[-1].upper()
            for start, resource in DOI_SUFFIX_RULES[prefix]:
                if suffix.startswith(start):
                    return resource
            return "doi"
        if prefix in DOI_PREFIXES:
            return DOI_PREFIXES[prefix]
        key = _norm(publisher)
        if key and key in ALIASES:
            return ALIASES[key]
        return None
    key = _norm(_PAREN_RE.sub(r"\1", scheme or "")) or _norm(publisher)
    return key or None


def describe(resource_slug: str, raw_label: str | None = None) -> Resource:
    """Label and category for a slug; an unknown slug is labelled with the raw name it came from
    and filed under Other, so the card still says something true."""
    known = RESOURCES.get(resource_slug)
    if known is not None:
        return known
    return Resource(resource_slug, raw_label or resource_slug, OTHER)


def is_reference_section(section: str | None) -> bool:
    return _norm(section).startswith(REFERENCE_SECTIONS)
