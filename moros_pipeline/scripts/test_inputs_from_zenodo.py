"""Hermetic tests for scripts/inputs_from_zenodo.py and scripts/runs_repo.py -- no network, no git remote.

What must never go wrong: an input that includes an already-enriched or non-positive record (it would be
re-enriched and the loader would refuse it), a row whose values differ from what the moros export
writes, a test input that silently drops a listed id, and a push to a repository not confirmed private.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import inputs_from_zenodo as iz  # noqa: E402
import runs_repo  # noqa: E402


def _doc(_id, journal="BMC bioinformatics", cls="positive", provider=None, title="T", abstract="A", year=2024):
    return {"_id": _id, "publication_metadata": {"title": title, "abstract": abstract, "journal": journal, "year": year},
            "llm_classification": {"classification": cls}, "llm_enrichment": {"provider": provider}}


def _dump(tmp_path, docs, name="dome-observatory-corpus-2026-09-25.jsonl.gz") -> Path:
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cohort_selects_only_unenriched_positives_with_text_in_the_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(iz, "load_cohort", lambda name: ["BMC bioinformatics", "Genome biology"])
    docs = [
        _doc("a"), _doc("b", journal="Genome biology", year=None),
        _doc("c", provider="deepseek"),            # already enriched
        _doc("d", cls="negative"),                 # not positive
        _doc("e", abstract="   "),                 # nothing to enrich from
        _doc("f", journal="Scientific reports"),   # outside the cohort
        _doc("g", journal="bmc bioinformatics"),   # names match exactly, never case-folded
    ]
    out = tmp_path / "runs" / "inputs"
    paths = iz.build_cohort(_dump(tmp_path, docs), "core", out, chunk_size=5000, zenodo_record="22967974")
    rows = _rows(paths[0])
    assert [r["record_id"] for r in rows] == ["a", "b"]
    assert list(rows[0]) == iz.OUTPUT_COLUMNS == ["record_id", "title", "abstract", "journal", "year"]
    assert rows[0]["year"] == "2024" and rows[1]["year"] == ""
    source = (out / "core" / "SOURCE.md").read_text()
    assert "22967974" in source and "2026-09-25" in source and "| Genome biology | 1 |" in source


def test_cohort_chunks_in_dump_order(tmp_path, monkeypatch):
    monkeypatch.setattr(iz, "load_cohort", lambda name: ["BMC bioinformatics"])
    paths = iz.build_cohort(_dump(tmp_path, [_doc(str(i)) for i in range(5)]), "core", tmp_path / "in", 2, "")
    assert [p.name for p in paths] == [f"enrich_input_core_zenodo-2026-09-25_c0{n}.csv" for n in (1, 2, 3)]
    assert [r["record_id"] for p in paths for r in _rows(p)] == ["0", "1", "2", "3", "4"]


def test_cohort_with_journal_order_runs_the_first_listed_journal_first(tmp_path, monkeypatch):
    monkeypatch.setattr(iz, "load_cohort", lambda name: ["Nature", "Scientific reports"])
    monkeypatch.setattr(iz, "load_cohort_order", lambda name: "journals")
    docs = [_doc("1", journal="Scientific reports"), _doc("2", journal="Nature"),
            _doc("3", journal="Scientific reports"), _doc("4", journal="Nature")]
    paths = iz.build_cohort(_dump(tmp_path, docs), "b2", tmp_path / "in", 3, "")
    assert [r["record_id"] for p in paths for r in _rows(p)] == ["2", "4", "1", "3"]
    assert "journal order" in (tmp_path / "in" / "b2" / "SOURCE.md").read_text()


def test_all_scope_reruns_e1_first_excludes_earlier_cohorts_and_puts_no_journal_last(tmp_path, monkeypatch):
    monkeypatch.setattr(iz, "load_cohort", lambda name: [])
    monkeypatch.setattr(iz, "load_cohort_order", lambda name: "reruns_then_journal_size")
    monkeypatch.setattr(iz, "load_cohort_scope",
                        lambda name: iz.Scope(all_journals=True, rerun=("e1",), exclude_cohorts=("core",)))
    out = tmp_path / "in"
    iz.write_csv(out / "core" / "c01.csv", [{"record_id": "done", "title": "", "abstract": "", "journal": "", "year": ""}])

    def enriched(_id, version, journal="Nature"):
        d = _doc(_id, journal=journal, provider="deepseek" if version == "e1" else "embl-ebi")
        d["llm_enrichment"]["prompt_version"] = version
        return d

    docs = [
        _doc("small", journal="Cell"),
        _doc("big1", journal="PloS one"), _doc("big2", journal="PloS one"),
        _doc("nojournal", journal=""),
        enriched("old", "e1"),                    # redone, and first
        enriched("current", "e3"),                # already on the latest prompt: never redone
        _doc("done", journal="PloS one"),         # in an earlier cohort's inputs
        _doc("neg", journal="Cell", cls="negative"),
    ]
    paths = iz.build_cohort(_dump(tmp_path, docs), "b3", out, 100, "")
    assert [r["record_id"] for r in _rows(paths[0])] == ["old", "big1", "big2", "small", "nojournal"]
    source = (out / "b3" / "SOURCE.md").read_text()
    assert "prompt e1 to redo (1 records" in source and "(no journal name) | 1 |" in source and "(1 ids)" in source


def test_exclusion_refuses_a_cohort_with_no_inputs(tmp_path):
    with pytest.raises(SystemExit, match="has no inputs"):
        iz.excluded_ids(tmp_path, ["core_bioinformatics"])


def test_unknown_order_is_refused(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"cohorts": {"odd": {"order": "random", "journals": ["A"]}}}))
    with pytest.raises(SystemExit, match="unknown order"):
        iz.load_cohort_order("odd", cfg)


def test_the_shipped_cohort_file_loads_and_has_no_duplicates():
    journals = iz.load_cohort("core_bioinformatics")
    assert len(journals) == 27 and "BMC bioinformatics" in journals
    assert iz.load_cohort_order("core_bioinformatics") == "dump"   # core's inputs rebuild byte-identical
    batch2 = iz.load_cohort("batch2_high_impact")
    assert iz.load_cohort_order("batch2_high_impact") == "journals"
    assert not set(batch2) & set(journals)                          # never re-enrich the core cohort
    assert batch2[:2] == ["Nature", "Science (New York, N.Y.)"]
    assert iz.load_cohort_scope("core_bioinformatics") == iz.Scope()             # unchanged defaults
    b3 = iz.load_cohort_scope("batch3_all_remaining")
    assert b3.all_journals and b3.rerun == ("e1",)
    assert set(b3.exclude_cohorts) == {"core_bioinformatics", "batch2_high_impact"}


def test_ids_mode_keeps_listed_order_and_refuses_a_missing_id(tmp_path):
    dump = _dump(tmp_path, [_doc("x", provider="deepseek"), _doc("y", cls="negative"), _doc("z")])
    ids = tmp_path / "manifest.csv"
    ids.write_text("record_id,cohort\ny,wave2\nx,wave2\nz,wave1\n")
    out = tmp_path / "input_wave2.csv"
    assert iz.build_ids(dump, ids, "wave2", out) == 2
    assert [r["record_id"] for r in _rows(out)] == ["y", "x"]   # whatever their state, in listed order
    ids.write_text("record_id,cohort\nmissing,wave2\n")
    with pytest.raises(SystemExit, match="not in the dump"):
        iz.build_ids(dump, ids, "wave2", out)


def test_sha256sums_cover_the_tracked_folders_only(tmp_path):
    (tmp_path / "inputs" / "c").mkdir(parents=True)
    (tmp_path / "inputs" / "c" / "a.csv").write_text("x")
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / ".gitkeep").write_text("")
    (tmp_path / "README.md").write_text("not summed")
    lines = runs_repo.write_sha256sums(tmp_path).read_text().splitlines()
    assert lines == [f"{runs_repo.sha256_file(tmp_path / 'inputs/c/a.csv')}  inputs/c/a.csv"]


def test_push_is_refused_without_a_token_or_a_private_answer(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="GH_TOKEN"):
        runs_repo.assert_private(tmp_path, token="")
    monkeypatch.setattr(runs_repo, "origin_slug", lambda repo: "someone/public-repo")

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"private": False}

    monkeypatch.setattr(runs_repo.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(SystemExit, match="not confirmed private"):
        runs_repo.assert_private(tmp_path, token="t")


def test_checkpoint_push_survives_a_remote_that_moved_with_no_git_identity(tmp_path, monkeypatch):
    """2026-11-17: a commit added on GitHub made `pull --rebase` replay the local checkpoint; with no git
    identity in the container the rebase stopped half-way and no checkpoint landed for 14 hours."""
    import subprocess

    def git(*a, cwd=tmp_path):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout

    monkeypatch.setenv("HOME", str(tmp_path / "home"))           # no global identity anywhere
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(k, raising=False)
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    git("init", "-q", "--bare", "-b", "main", "remote.git")
    git("clone", "-q", "remote.git", "laptop")
    laptop = tmp_path / "laptop"
    (laptop / "events").mkdir()
    (laptop / "events" / "a.csv").write_text("1\n")
    git(*ident, "add", ".", cwd=laptop)
    git(*ident, "commit", "-q", "-m", "init", cwd=laptop)
    git("push", "-q", "origin", "main", cwd=laptop)
    git("clone", "-q", "remote.git", "web")                      # the remote moves (e.g. the API adds a file)
    (tmp_path / "web" / "wf.yml").write_text("x\n")
    git(*ident, "add", ".", cwd=tmp_path / "web")
    git(*ident, "commit", "-q", "-m", "workflow", cwd=tmp_path / "web")
    git("push", "-q", cwd=tmp_path / "web")

    monkeypatch.setattr(runs_repo, "assert_private", lambda repo: None)
    (laptop / "events" / "a.csv").write_text("1\n2\n")
    runs_repo.write_sha256sums(laptop)
    assert runs_repo.commit_and_push(laptop, "checkpoint") is True
    assert not (laptop / ".git" / "rebase-merge").exists()
    assert git("log", "--format=%s", "main", cwd=tmp_path / "remote.git").split() == ["checkpoint", "workflow", "init"]
