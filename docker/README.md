# Docker images

- **`Dockerfile.cpu`** (this pass): everything Phase 0-1 needs -- ingest, dedupe, fulltext
  manifest, TF-IDF/KeyBERT keyword extraction, and the Streamlit curation app. Runs on the
  laptop's CPU; no GPU required.

- **`Dockerfile.gpu`** (not built yet, Phase 4): will add a CUDA-enabled PyTorch base image for
  fine-tuning Bioformer/PubMedBERT on the lab NVIDIA GPU. Deferred until Phase 4 starts, since the
  lab GPU's CUDA/driver version needs confirming first (see ROADMAP.md).

- **An Ollama-backed LLM service** (not built yet, Phase 5): will add a service running one or
  more local LLM backends for the bake-off described in ROADMAP.md Phase 5, alongside cloud API
  backends that don't need a container at all. Deferred until Phase 5 starts.

Both `curate` and `pipeline` services in `docker-compose.yml` bind-mount
`/home/gavinfarrell/PhD_Code` read-only at the same absolute path inside the container, because
`configs/sources.yaml` references the sibling repos (`DOME_Top_Curate`,
`DOME-Copilot-Data-Analysis`, etc.) by absolute host path. On a machine without those sibling
repos present, only the sources that need them (`pdf_directory_gold` label sources and all
`fulltext_roots`) will be unavailable -- the CSV/TSV/JSON sources still load fine, and
`dome-triage fulltext fetch --pmcid ...` provides an independent full-text fallback (see
AGENTS.md).
