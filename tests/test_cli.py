from typer.testing import CliRunner

import ragfoundry.pipeline as pipeline_module
from ragfoundry.cli import app

from .conftest import KeywordEmbeddings, rule_llm


def test_init_ingest_query_sources(tmp_path, corpus, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_module, "build_embeddings", lambda config: KeywordEmbeddings())
    monkeypatch.setattr(
        pipeline_module, "build_chat_model", lambda config: rule_llm(lambda p: "Answer [1]")
    )
    runner = CliRunner()
    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["init"]).exit_code == 1

    result = runner.invoke(app, ["ingest", str(corpus)])
    assert result.exit_code == 0, result.output
    assert "Added: 4" in result.output

    result = runner.invoke(app, ["query", "solar panels"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Answer [1]")

    result = runner.invoke(app, ["sources"])
    assert "solar.md" in result.output

    result = runner.invoke(app, ["ingest", str(tmp_path / "missing.txt")])
    assert result.exit_code == 1
    assert "Source not found" in result.output
