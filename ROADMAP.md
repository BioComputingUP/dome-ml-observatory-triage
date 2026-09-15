# Roadmap

1. **Automated Zenodo bulk push.** Monthly GitHub Actions workflow: cursor-loop `GET /api/export`,
   gzip, deposit through the Zenodo API with a sidecar (count, size, sha256, `schema_version`) and
   the schema release. Reuse `DOME_zenodo_archive/download_dome_registry.py`, `ZENODO_TOKEN` from
   Actions secrets. Then add the deposit to that month's release metadata as a Zenodo distribution
   with its DOI (`build_release_metadata.py`, [docs/release_metadata.md](docs/release_metadata.md)),
   and replace the unregistered DOI hardcoded on `/download/bulk`.

2. **Keep the two repositories aligned.** Last, once the item above lands: `check_alignment.py`
   clean, and no claim, count or link in either repo's `README.md`, `ROADMAP.md` or skills
   contradicting the other's. The sister roadmap carries the matching item.
