# docker/

`Dockerfile.cpu` is the only image, and `docker-compose.yml`'s single `pipeline` service builds
from it. CPU only. It runs the classification and enrichment CLI and nothing else; no container
here ever opens a connection to the database.

It bakes the NLTK corpora and the KeyBERT/sentence-transformers weights in at build time, so a
run needs no network access to tokenise, and so a host-Python run cannot silently diverge from
what the container does. That is why `AGENTS.md` requires Docker for the engine.

What it mounts, and why each one is needed:

| Mount | Why |
|---|---|
| `./data` | the spend and calibration logs the budget check reads and appends to |
| `./configs` | the paths and the budget cap |
| `./.git` (read-only) | so the provenance ledger can record the commit a run happened at |
| `./curation_criteria` (read-only) | `CRITERIA.md` and the three vocabularies, read at runtime to build the prompts |
| `./moros_pipeline` (read-write) | a run reads its staged or exported CSV from here and writes its event log back beside it |

Rebuilding leaves the previous image dangling at about seven gigabytes. Run `docker image prune -f`
after a few rebuilds and check `df -h /`. Never `docker system prune -a` without asking.
