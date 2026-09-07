from typer.testing import CliRunner

from dome_triage.cli import app

runner = CliRunner()


def test_top_level_help_lists_all_subcommand_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("ingest", "dedupe", "fulltext", "keywords", "curate", "pipeline", "llm-classify"):
        assert group in result.output


def test_llm_classify_help_lists_expected_commands():
    result = runner.invoke(app, ["llm-classify", "--help"])
    assert result.exit_code == 0
    for command in (
        "validate-criteria", "calibrate", "project-cost", "sample", "classify",
        "calibrate-fallback", "run-fallback",
    ):
        assert command in result.output


def test_reporting_help_lists_second_curator_commands():
    result = runner.invoke(app, ["reporting", "--help"])
    assert result.exit_code == 0
    assert "agreement" in result.output
    assert "profile-second-curator" in result.output


def test_curate_help_lists_materialize_cross_curate_resolutions():
    result = runner.invoke(app, ["curate", "--help"])
    assert result.exit_code == 0
    assert "materialize-cross-curate-resolutions" in result.output


def test_llm_classify_classify_requires_estimated_usd():
    """classify must never invent its own cost figure -- omitting --estimated-usd is a hard error,
    not a silent default."""
    result = runner.invoke(app, ["llm-classify", "classify", "--tier", "flash"])
    assert result.exit_code != 0


def test_ingest_help_lists_expected_commands():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "load-sources" in result.output
    assert "enrich-metadata" in result.output
    assert "fetch-clear-negatives" in result.output
    assert "screen-clear-negatives" in result.output
    assert "merge-strong-negatives" in result.output
    assert "backfill-canonical-metadata" in result.output


def test_keywords_help_lists_expected_commands():
    result = runner.invoke(app, ["keywords", "--help"])
    assert result.exit_code == 0
    assert "build-lexicon" in result.output
    assert "materialize-lexicon" in result.output
    assert "seed-additional-terms" in result.output
    assert "suggest-final-lexicon" in result.output
    assert "scoring-bakeoff" in result.output


def test_pipeline_run_rejects_unknown_step():
    result = runner.invoke(app, ["pipeline", "run", "--steps", "not-a-real-step"])
    assert result.exit_code != 0
