#!/usr/bin/env python3
"""Archives the whole corpus to Zenodo as a new version of one record, after a load.

The corpus is read from moros (read-only) and written as gzipped JSON Lines -- one complete
document per line, the same documents `GET /api/export` serves -- with a sidecar,
`archive-metadata.json` (counts, sha256 and md5 of every file, schema version, the latest
classification and enrichment timestamps), and the schema release the documents follow. Each run
publishes a new version of the Observatory's Zenodo record, so the concept DOI
10.5281/zenodo.22259905 always resolves to the latest corpus and every earlier one stays citable.

A step of the refresh cycle (`refresh-cycle` skill, after the post-load checklist), not a
schedule: the corpus only changes when something is loaded, so that is when it is archived. A run
whose live figures match the latest version's sidecar stops there, so an unnecessary run is
harmless; `--force` archives anyway.

Order is what keeps a failure clean: nothing is written to Zenodo until the export is complete,
counted and hashed locally. A version draft this run created is deleted if anything after it fails
(`--keep-draft` keeps it for inspection). A draft already open on Zenodo is never reused or deleted
-- it may be a hand edit -- so the run stops and says so. A published version is never touched.

Access: a version inherits the record's access. An embargo whose date has passed becomes open. A
record still `restricted` (the placeholder state) refuses to publish until `--embargo-until` or
`--open` says what the files should be.

Needs `ZENODO_TOKEN` (deposit:write and deposit:actions) in the root `.env` or the environment, the
VPN and `moros_pipeline/.env`, and the sibling dome-ml-observatory checkout for the schema file
(`DOME_OBSERVATORY_DIR`, default `../dome-ml-observatory`).

    python3 scripts/zenodo_archive.py --check                    # compare live figures to the latest version
    python3 scripts/zenodo_archive.py --dry-run --limit 1000     # export + sidecar locally, no Zenodo writes
    python3 scripts/zenodo_archive.py --no-publish               # full run, stop at an unpublished draft
    python3 scripts/zenodo_archive.py --publish-draft <id>       # ...then publish it once checked
    python3 scripts/zenodo_archive.py                            # full run, publish
    python3 scripts/zenodo_archive.py --embargo-until 2026-10-25 # first real version: files open on that date
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "moros_pipeline" / "scripts"))

from moros_client import Moros  # noqa: E402

ZENODO = "https://zenodo.org"
BASE_RECORD_ID = 22259906  # the record's first version; its concept is 22259905
SIDECAR = "archive-metadata.json"  # a stable name: the next run's change check reads it back
OUT_DIR = REPO / "moros_pipeline" / "output" / "zenodo"
OBSERVATORY = "https://observatory.dome-ml.org"


# -- configuration --------------------------------------------------------------------------------


def zenodo_token() -> str:
    """ZENODO_TOKEN from the environment, else the root `.env` -- the same minimal KEY=VALUE reading
    as moros_client.load_env, so no python-dotenv."""
    token = os.environ.get("ZENODO_TOKEN", "")
    env = REPO / ".env"
    if not token and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().partition("=")
            if key.strip() == "ZENODO_TOKEN":
                token = value.strip()
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set -- add it to the root .env (see .env.example)")
    return token


def observatory_dir() -> Path:
    return Path(os.environ.get("DOME_OBSERVATORY_DIR", REPO.parent / "dome-ml-observatory"))


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# -- Zenodo ---------------------------------------------------------------------------------------


class Zenodo:
    """The legacy deposit API, which is what versioning and bucket uploads still go through. The
    token travels in a header only, never a URL, so it cannot end up in a log."""

    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def _check(self, resp: requests.Response, what: str) -> requests.Response:
        if resp.status_code in (401, 403):
            raise SystemExit(f"Zenodo refused {what} ({resp.status_code}): the token is missing a "
                             f"scope or has been revoked -- make a new one with deposit:write and "
                             f"deposit:actions, and put it in .env")
        if not resp.ok:
            raise RuntimeError(f"Zenodo {what} failed: {resp.status_code} {resp.text[:400]}")
        return resp

    def get(self, path: str, what: str, **params) -> dict:
        return self._check(self.session.get(f"{self.base}{path}", params=params, timeout=60), what).json()

    def deposition(self, dep_id: int | str) -> dict:
        return self.get(f"/api/deposit/depositions/{dep_id}", f"reading deposition {dep_id}")

    def versions(self, concept: str) -> list[dict]:
        return self.get("/api/deposit/depositions", "listing versions",
                        q=f"conceptrecid:{concept}", all_versions="true", size=100)

    def latest_published(self, concept: str) -> dict:
        published = [d for d in self.versions(concept) if d.get("submitted")]
        if not published:
            raise SystemExit(f"no published version of concept {concept} -- nothing to version from")
        return max(published, key=lambda d: int(d["id"]))

    def read_sidecar(self, dep: dict) -> dict | None:
        for f in dep.get("files") or []:
            if f.get("filename") == SIDECAR:
                resp = self._check(self.session.get(f["links"]["download"], timeout=60), "reading the sidecar")
                return resp.json()
        return None

    def delete_draft(self, dep_id: int | str) -> None:
        self._check(self.session.delete(f"{self.base}/api/deposit/depositions/{dep_id}", timeout=60),
                    f"deleting draft {dep_id}")

    def new_version(self, dep_id: int | str) -> dict:
        resp = self._check(self.session.post(
            f"{self.base}/api/deposit/depositions/{dep_id}/actions/newversion", timeout=120), "creating a version")
        return self._check(self.session.get(resp.json()["links"]["latest_draft"], timeout=60),
                           "reading the new draft").json()

    def clear_files(self, draft: dict) -> None:
        for f in self.get(f"/api/deposit/depositions/{draft['id']}/files", "listing draft files"):
            self._check(self.session.delete(f["links"]["self"], timeout=60), f"removing {f['filename']}")

    def upload(self, draft: dict, path: Path, md5: str) -> None:
        with path.open("rb") as body:
            resp = self._check(self.session.put(f"{draft['links']['bucket']}/{path.name}", data=body,
                                                timeout=None), f"uploading {path.name}")
        got = (resp.json().get("checksum") or "").removeprefix("md5:")
        if got != md5:
            raise RuntimeError(f"{path.name}: Zenodo stored md5 {got}, the local file is {md5}")

    def set_metadata(self, draft: dict, metadata: dict) -> None:
        self._check(self.session.put(f"{self.base}/api/deposit/depositions/{draft['id']}",
                                     json={"metadata": metadata}, timeout=60), "writing metadata")

    def publish(self, draft: dict) -> dict:
        return self._check(self.session.post(
            f"{self.base}/api/deposit/depositions/{draft['id']}/actions/publish", timeout=300), "publishing").json()


# -- the corpus -----------------------------------------------------------------------------------


def live_figures(moros: Moros) -> dict:
    """What a new version would say about the corpus, read off moros -- also the change check."""
    c = moros.collection
    by_class = {r["_id"]: r["n"] for r in c.aggregate(
        [{"$group": {"_id": "$llm_classification.classification", "n": {"$sum": 1}}}])}
    latest = next(c.aggregate([{"$group": {"_id": None,
                                           "c": {"$max": "$llm_classification.timestamp"},
                                           "e": {"$max": "$llm_enrichment.timestamp"}}}]), {})
    versions = c.distinct("schema_version")
    if len(versions) != 1:
        raise SystemExit(f"documents carry schema versions {versions} -- finish that migration first")
    return {
        "record_count": sum(by_class.values()),
        "classification_counts": {k or "unclassified": v for k, v in sorted(by_class.items(), key=lambda kv: str(kv[0]))},
        "schema_version": versions[0],
        "last_classification": latest.get("c"),
        "last_enrichment": latest.get("e"),
    }


CHANGE_KEYS = ("record_count", "schema_version", "last_classification", "last_enrichment")


def unchanged(live: dict, sidecar: dict | None) -> bool:
    return sidecar is not None and all(sidecar.get(k) == live[k] for k in CHANGE_KEYS)


class _Hashing:
    """A write-through file wrapper that counts and hashes what passes through it."""

    def __init__(self, raw) -> None:
        self.raw, self.size = raw, 0
        self.sha256, self.md5 = hashlib.sha256(), hashlib.md5()

    def write(self, data: bytes) -> int:
        self.size += len(data)
        self.sha256.update(data)
        self.md5.update(data)
        return self.raw.write(data)

    def flush(self) -> None:
        self.raw.flush()


def export(moros: Moros, path: Path, limit: int | None) -> dict:
    """Every document, `_id` order, one JSON object per line, gzipped with mtime 0 so the same
    corpus always produces the same bytes. Hashes both the lines and the compressed file as they
    are written: the 3 GB of uncompressed JSON never exists on disk."""
    lines = hashlib.sha256()
    n = raw_bytes = 0
    cursor = moros.collection.find({}, sort=[("_id", 1)], batch_size=2_000, limit=limit or 0)
    with path.open("wb") as out:
        hashed = _Hashing(out)
        with gzip.GzipFile(filename="", mode="wb", fileobj=hashed, mtime=0, compresslevel=6) as gz:
            for doc in cursor:
                line = (json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode()
                gz.write(line)
                lines.update(line)
                raw_bytes += len(line)
                n += 1
                if n % 100_000 == 0:
                    print(f"  exported {n:,}", flush=True)
    return {"records": n, "bytes": hashed.size, "sha256": hashed.sha256.hexdigest(),
            "md5": hashed.md5.hexdigest(), "uncompressed_bytes": raw_bytes,
            "uncompressed_sha256": lines.hexdigest()}


def file_digest(path: Path) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


# -- the record's metadata ------------------------------------------------------------------------


def access_fields(current: dict, embargo_until: date | None, open_now: bool, today: date) -> dict:
    """The access fields for the new version. Explicit flags win; otherwise the version inherits the
    record's access, with a lapsed embargo becoming open. `restricted` is never inherited silently:
    it is the placeholder state, and publishing a corpus behind it would hide it with no end date."""
    if open_now:
        return {"access_right": "open"}
    if embargo_until:
        if embargo_until <= today:
            raise SystemExit(f"--embargo-until {embargo_until} is not in the future")
        return {"access_right": "embargoed", "embargo_date": embargo_until.isoformat()}
    right = current.get("access_right")
    if right == "embargoed":
        until = date.fromisoformat(current["embargo_date"])
        return ({"access_right": "embargoed", "embargo_date": until.isoformat()} if until > today
                else {"access_right": "open"})
    if right == "open":
        return {"access_right": "open"}
    raise SystemExit(f"the record is {right!r} -- say what the files should be: --embargo-until "
                     f"YYYY-MM-DD, or --open")


def description(fig: dict, schema_file: str) -> str:
    counts = fig["classification_counts"]
    return (
        f"<p>The DOME Observatory corpus as of {fig['version']}: {fig['record_count']:,} publications "
        f"from Europe PMC, each screened for whether it is an AI/ML methods paper "
        f"({counts.get('positive', 0):,} positive, {counts.get('negative', 0):,} negative, "
        f"{counts.get('undeterminable', 0):,} undeterminable), with the rationale for each verdict, "
        f"controlled-vocabulary tags where the enrichment pass has run, citation counts and links "
        f"to the data each paper cites. Records are LLM-classified by a method validated against a "
        f"hand-annotated expert benchmark; they are not individually curator-reviewed.</p>"
        f"<p>Files: the records as JSON Lines, one complete record per line, gzipped -- the same "
        f"records <a href=\"{OBSERVATORY}/api/export\">{OBSERVATORY}/api/export</a> serves; "
        f"<code>{SIDECAR}</code>, with the counts and the sha256 of every file; and "
        f"<code>{schema_file}</code>, the record schema (v{fig['schema_version']}) the records "
        f"follow. Each version of this record is one corpus release; the concept DOI always "
        f"resolves to the latest.</p>"
        f"<p>Licence: the screening decisions, rationales, enrichment tags, vocabularies, schema and "
        f"statistics this project adds are CC BY 4.0. The bibliographic metadata and abstracts "
        f"each record carries were not created by this project and are not covered by it; most "
        f"come from Europe PMC, which should be cited alongside this dataset.</p>"
        f"<p>Browse and search: <a href=\"{OBSERVATORY}\">{OBSERVATORY}</a></p>"
    )


def version_metadata(current: dict, fig: dict, access: dict, schema_file: str) -> dict:
    """The draft's metadata with only what a release changes: the version, date, description,
    access and the links back. Creators, licence, title and anything added by hand on Zenodo stay
    as they are."""
    metadata = {k: v for k, v in current.items()
                if k not in ("doi", "prereserve_doi", "access_right", "embargo_date", "access_conditions")}
    # The API reads creators back with `"affiliation": null`, and refuses a null it was sent.
    metadata["creators"] = [{k: v for k, v in c.items() if v is not None}
                            for c in metadata.get("creators") or []]
    related = [r for r in metadata.get("related_identifiers") or []
               if r.get("identifier") not in (OBSERVATORY, f"{OBSERVATORY}/api/export")]
    metadata.update({
        "title": current.get("title") or "DOME Observatory",
        "upload_type": "dataset",
        "version": fig["version"],
        "publication_date": fig["version"],
        "description": description(fig, schema_file),
        "keywords": sorted(set(metadata.get("keywords") or []) | {
            "machine learning", "artificial intelligence", "life sciences", "Europe PMC",
            "research metadata", "DOME"}),
        "related_identifiers": related + [
            {"identifier": OBSERVATORY, "relation": "isSupplementTo", "resource_type": "other"},
            {"identifier": f"{OBSERVATORY}/api/export", "relation": "isIdenticalTo", "resource_type": "dataset"},
        ],
        **access,
    })
    return metadata


# -- the run --------------------------------------------------------------------------------------


def publish_draft(zen: Zenodo, draft_id: int, concept: str) -> None:
    """Publishes a draft a `--no-publish` run left for inspection, without exporting again."""
    draft = zen.deposition(draft_id)
    if draft.get("submitted") or str(draft.get("conceptrecid")) != str(concept):
        raise SystemExit(f"{draft_id} is not an unpublished version draft of concept {concept}")
    names = {f.get("filename") for f in draft.get("files") or []}
    if SIDECAR not in names or not any(n.endswith(".jsonl.gz") for n in names):
        raise SystemExit(f"draft {draft_id} holds {sorted(names)} -- not an archive run's upload")
    published = zen.publish(draft)
    print(f"zenodo_archive: published version {published['id']} -- DOI {published.get('doi')}, "
          f"concept DOI {published.get('conceptdoi')}")


def run(args: argparse.Namespace) -> None:
    today = datetime.now(timezone.utc).date()
    zen = Zenodo(args.zenodo_url, zenodo_token())
    base = zen.deposition(args.record_id)  # fails in seconds on a bad token, not after the export
    concept = base["conceptrecid"]
    if args.publish_draft:
        publish_draft(zen, args.publish_draft, concept)
        return
    latest = zen.latest_published(concept) if base.get("submitted") else base
    sidecar = zen.read_sidecar(latest)

    # zlib on the wire: the export reads the whole collection over the VPN, which is
    # bandwidth-bound -- about 35 minutes compressed against 110 plain (moros_client.Moros).
    with Moros.from_env(compressors="zlib") as moros:
        print(f"zenodo_archive: {moros.describe()}")
        fig = live_figures(moros)
        fig["version"] = today.isoformat()
        print("zenodo_archive: live " + ", ".join(f"{k}={fig[k]}" for k in CHANGE_KEYS))
        if sidecar:
            print(f"zenodo_archive: latest version {latest['id']} ({latest['metadata'].get('version')}) "
                  + ", ".join(f"{k}={sidecar.get(k)}" for k in CHANGE_KEYS))
        else:
            print(f"zenodo_archive: version {latest['id']} carries no {SIDECAR} -- nothing archived yet")
        if unchanged(fig, sidecar) and not args.force:
            print("zenodo_archive: unchanged since the latest version -- nothing to do (--force to archive anyway)")
            return
        if args.check:
            print("zenodo_archive: changed -- a run would archive a new version")
            return
        access = access_fields(latest["metadata"], args.embargo_until, args.open, today)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        corpus = OUT_DIR / f"dome-observatory-corpus-{fig['version']}.jsonl.gz"
        print(f"zenodo_archive: exporting to {corpus}" + (f" (--limit {args.limit})" if args.limit else ""))
        exported = export(moros, corpus, args.limit)

    if not args.limit and exported["records"] != fig["record_count"]:
        raise SystemExit(f"exported {exported['records']:,} records but counted {fig['record_count']:,} -- "
                         f"the corpus changed during the export; run again once the load is done")
    schema_src = observatory_dir() / "schema" / "releases" / f"v{fig['schema_version']}" / "ai-ml-landscape.schema.json"
    if not schema_src.exists():
        raise SystemExit(f"{schema_src} not found -- set DOME_OBSERVATORY_DIR to the dome-ml-observatory checkout")
    schema_file = OUT_DIR / f"ai-ml-landscape.schema.v{fig['schema_version']}.json"
    schema_file.write_bytes(schema_src.read_bytes())

    record = {**fig, "exported_at": datetime.now(timezone.utc).isoformat(),
              "source": "dome_observatory.Content, read-only",
              "generated_by": f"dome-ml-observatory-triage scripts/zenodo_archive.py @ {git_head()}",
              "format": "JSON Lines, UTF-8, gzip: one complete record per line, _id order, as GET /api/export serves them",
              "files": {corpus.name: {k: exported[k] for k in ("bytes", "sha256", "md5")},
                        schema_file.name: file_digest(schema_file)},
              "uncompressed": {"bytes": exported["uncompressed_bytes"], "sha256": exported["uncompressed_sha256"]}}
    if args.limit:
        record["limit"] = args.limit  # a trial export: never mistaken for the corpus
    sidecar_path = OUT_DIR / SIDECAR
    sidecar_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"zenodo_archive: {exported['records']:,} records, {exported['bytes'] / 1e6:,.1f} MB gzipped "
          f"({exported['uncompressed_bytes'] / 1e6:,.1f} MB raw), sha256 {exported['sha256'][:16]}...")

    if args.dry_run:
        print(f"zenodo_archive: --dry-run -- files are in {OUT_DIR}; nothing was sent to Zenodo")
        return
    if args.limit and not args.force:
        raise SystemExit("--limit exports a sample; only --dry-run may use it (or --force to deposit a sample)")

    # Zenodo writes start here, only after everything above succeeded. A version draft already open
    # is never reused or deleted: it may be someone's hand edit, not an interrupted run.
    pending = (latest.get("links") or {}).get("latest_draft", "")
    if base.get("submitted") and pending and not pending.rstrip("/").endswith(f"/{latest['id']}"):
        raise SystemExit(f"a version draft is already open ({pending}) -- publish or discard it on "
                         f"Zenodo, then run again")
    created = base.get("submitted")
    draft = zen.new_version(latest["id"]) if created else base
    try:
        zen.clear_files(draft)
        for path, digest in ((corpus, exported["md5"]), (sidecar_path, file_digest(sidecar_path)["md5"]),
                             (schema_file, record["files"][schema_file.name]["md5"])):
            print(f"zenodo_archive: uploading {path.name}")
            zen.upload(draft, path, digest)
        zen.set_metadata(draft, version_metadata(latest["metadata"], fig, access, schema_file.name))
        if args.no_publish:
            print(f"zenodo_archive: --no-publish -- draft ready at {ZENODO}/uploads/{draft['id']}")
            return
        published = zen.publish(draft)
    except BaseException:
        if created and not args.keep_draft:
            print(f"zenodo_archive: failed -- deleting draft {draft['id']} (--keep-draft to keep it)")
            zen.delete_draft(draft["id"])
        raise
    print(f"zenodo_archive: published version {published['id']} -- DOI {published.get('doi')}, "
          f"concept DOI {published.get('conceptdoi')}, {access}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Compare the live figures with the latest version, then stop.")
    parser.add_argument("--dry-run", action="store_true", help="Export and write the sidecar locally; send nothing to Zenodo.")
    parser.add_argument("--no-publish", action="store_true", help="Upload into a version draft and stop before publishing.")
    parser.add_argument("--keep-draft", action="store_true", help="Keep a draft this run created if a later step fails.")
    parser.add_argument("--force", action="store_true", help="Archive even when nothing changed since the latest version.")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N documents (with --dry-run).")
    parser.add_argument("--embargo-until", type=date.fromisoformat, default=None,
                        help="Publish with the files embargoed until this date (YYYY-MM-DD).")
    parser.add_argument("--open", action="store_true", help="Publish with the files open now.")
    parser.add_argument("--publish-draft", type=int, default=None, metavar="ID",
                        help="Publish the draft a --no-publish run left, once it has been checked.")
    parser.add_argument("--record-id", type=int, default=BASE_RECORD_ID)
    parser.add_argument("--zenodo-url", default=ZENODO, help="https://sandbox.zenodo.org for a rehearsal.")
    args = parser.parse_args()
    if args.embargo_until and args.open:
        parser.error("--embargo-until and --open contradict each other")
    run(args)


if __name__ == "__main__":
    main()
