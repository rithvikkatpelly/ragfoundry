import pytest
from pydantic import ValidationError

from ragfoundry import ConfigurationError, RAGConfig


def test_yaml_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-secret")
    path = tmp_path / "rag.yaml"
    path.write_text(
        "llm:\n  provider: openai\n  model: gpt-test\n  api_key: ${TEST_OPENAI_KEY}\n"
        "vector_store:\n  provider: qdrant\n  url: ${QDRANT_URL:-http://localhost:6333}\n"
    )
    config = RAGConfig.from_yaml(path)
    assert config.llm.api_key.get_secret_value() == "sk-secret"
    assert config.vector_store.url == "http://localhost:6333"
    assert "sk-secret" not in repr(config)


def test_unset_environment_variable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    path = tmp_path / "rag.yaml"
    path.write_text("llm:\n  provider: groq\n  model: m\n  api_key: ${NOT_SET_ANYWHERE}\n")
    with pytest.raises(ConfigurationError, match="NOT_SET_ANYWHERE"):
        RAGConfig.from_yaml(path)


@pytest.mark.parametrize(
    "data",
    [
        {"chunking": {"chunk_size": 100, "chunk_overlap": 100}},
        {"retrieval": {"top_k": 10, "fetch_k": 5}},
        {"vector_store": {"provider": "redis"}},
        {"llm": {"provider": "openai", "model": "m", "temprature": 0.1}},
    ],
)
def test_invalid_settings_are_rejected(data):
    with pytest.raises(ValidationError):
        RAGConfig.model_validate(data)
