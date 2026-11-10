"""Factories build the right client for each provider, with keys passed through."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ragfoundry import MissingDependencyError, RAGConfig, RAGPipeline
from ragfoundry._utils import require
from ragfoundry.config import EmbeddingConfig, LLMConfig, ObservabilityConfig
from ragfoundry.models import build_chat_model, build_embeddings
from ragfoundry.observability import build_observer

from .conftest import KeywordEmbeddings, rule_llm


@pytest.mark.parametrize(
    ("provider", "class_name", "secret_field"),
    [
        ("groq", "ChatGroq", "groq_api_key"),
        ("openai", "ChatOpenAI", "openai_api_key"),
        ("gemini", "ChatGoogleGenerativeAI", "google_api_key"),
        ("claude", "ChatAnthropic", "anthropic_api_key"),
        ("ollama", "ChatOllama", None),
    ],
)
def test_chat_models(provider, class_name, secret_field):
    config = LLMConfig(provider=provider, model="m", api_key="k-123", max_tokens=50)
    model = build_chat_model(config)
    assert type(model).__name__ == class_name
    if secret_field:
        assert getattr(model, secret_field).get_secret_value() == "k-123"
    assert "temperature" not in {k for k, v in model.model_dump().items() if v is not None} or (
        getattr(model, "temperature", None) in (None, 0.7, 1.0)
    )


@pytest.mark.parametrize(
    ("provider", "class_name"),
    [("openai", "OpenAIEmbeddings"), ("gemini", "GoogleGenerativeAIEmbeddings")],
)
def test_hosted_embeddings(provider, class_name):
    model = build_embeddings(EmbeddingConfig(provider=provider, model="m", api_key="k-123"))
    assert type(model).__name__ == class_name


def test_missing_extra_names_the_install_command():
    with pytest.raises(MissingDependencyError, match=r"pip install 'ragfoundry\[x\]'"):
        require("module_that_does_not_exist", "x")


class _Collector(BaseHTTPRequestHandler):
    requests: list = []

    def _record(self):
        length = int(self.headers.get("content-length", 0))
        self.requests.append((self.command, self.path, self.rfile.read(length)))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({}).encode())

    do_POST = do_PATCH = do_PUT = do_GET = _record

    def log_message(self, *args):
        pass


@pytest.fixture
def collector():
    _Collector.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _Collector.requests
    server.shutdown()


def _traced_query(tmp_path, corpus, observability):
    config = RAGConfig.model_validate(
        {
            "persist_dir": str(tmp_path),
            "vector_store": {"provider": "faiss"},
            "observability": observability,
        }
    )
    pipeline = RAGPipeline(config, llm=rule_llm(lambda p: "ok"), embeddings=KeywordEmbeddings())
    pipeline.ingest(corpus)
    pipeline.query("solar")
    pipeline.close()


def test_langsmith_sends_runs(tmp_path, corpus, collector):
    url, requests = collector
    _traced_query(tmp_path, corpus, {"provider": "langsmith", "api_key": "ls-key", "endpoint": url})
    bodies = b" ".join(body for method, path, body in requests if "runs" in path)
    assert b"ragfoundry.query" in bodies
    assert b"ragfoundry.ingest" in bodies


def test_langfuse_exports_spans(tmp_path, corpus, collector):
    url, requests = collector
    _traced_query(
        tmp_path,
        corpus,
        {"provider": "langfuse", "public_key": "pk-test", "secret_key": "sk-test", "endpoint": url},
    )
    paths = [path for method, path, body in requests]
    assert any("/otel/v1/traces" in path for path in paths), paths


def test_tracing_requires_credentials():
    from ragfoundry import ConfigurationError

    with pytest.raises(ConfigurationError, match="api_key"):
        build_observer(ObservabilityConfig(provider="langsmith"))
    with pytest.raises(ConfigurationError, match="secret_key"):
        build_observer(ObservabilityConfig(provider="langfuse", public_key="pk"))
