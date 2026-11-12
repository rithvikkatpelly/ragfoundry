"""Tests against real services and models.

Run separately with ``pytest -m integration``. They are excluded from the default run because
faiss-cpu and torch cannot share a process on macOS, and the unit tests use FAISS.
Each test skips when its service is unreachable:
- pgvector:  RF_PG_URL      (default postgresql+psycopg://rf:rf@localhost:55432/rf)
- Weaviate:  RF_WEAVIATE_URL (default http://localhost:58080, gRPC on RF_WEAVIATE_GRPC=55051)
- Ollama:    RF_OLLAMA_MODEL (default qwen3:8b) on the default Ollama port
"""

import os
import socket
from urllib.parse import urlparse

import pytest

from ragfoundry import EvalSample, RAGConfig, RAGPipeline

from .conftest import KeywordEmbeddings

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("RF_PG_URL", "postgresql+psycopg://rf:rf@localhost:55432/rf")
WEAVIATE_URL = os.environ.get("RF_WEAVIATE_URL", "http://localhost:58080")
WEAVIATE_GRPC = int(os.environ.get("RF_WEAVIATE_GRPC", "55051"))
OLLAMA_MODEL = os.environ.get("RF_OLLAMA_MODEL", "qwen3:8b")
HF_EMBEDDINGS = {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}


def _reachable(url: str, default_port: int) -> bool:
    parts = urlparse(url)
    try:
        socket.create_connection((parts.hostname, parts.port or default_port), timeout=1).close()
    except OSError:
        return False
    return True


def _ollama_ready() -> bool:
    if not _reachable("http://localhost:11434", 11434):
        return False
    import httpx

    tags = httpx.get("http://localhost:11434/api/tags", timeout=5).json()
    return any(m["name"] == OLLAMA_MODEL for m in tags.get("models", []))


def _ollama_llm() -> dict:
    # Disabling reasoning keeps qwen3 replies short; it is passed through ``options``.
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "temperature": 0,
        "options": {"reasoning": False},
    }


def _store_roundtrip(tmp_path, corpus, vector_store):
    config = RAGConfig.model_validate({"persist_dir": str(tmp_path), "vector_store": vector_store})
    pipeline = RAGPipeline(config, embeddings=KeywordEmbeddings())
    try:
        assert pipeline.ingest(corpus).chunks_added == 4
        assert pipeline.ingest(corpus).chunks_added == 0
        top = pipeline.retrieve("solar panel efficiency", top_k=1)[0]
        assert top.metadata["source"].endswith("solar.md")
        filtered = pipeline.retrieve("panels", where={"file_type": "csv"})
        assert filtered and {d.metadata["file_type"] for d in filtered} == {"csv"}
        assert pipeline.delete_source(str(corpus / "solar.md")) == 1
        remaining = pipeline.retrieve("solar panel efficiency", top_k=4)
        assert not any(d.metadata["source"].endswith("solar.md") for d in remaining)
    finally:
        pipeline.close()


def test_milvus_lite(tmp_path, corpus):
    pytest.importorskip("milvus_lite")
    _store_roundtrip(tmp_path, corpus, {"provider": "milvus", "collection_name": "rf_test"})


@pytest.mark.skipif(not _reachable(PG_URL.replace("+psycopg", ""), 5432), reason="no Postgres")
def test_pgvector(tmp_path, corpus):
    name = f"rf_{os.getpid()}_{tmp_path.name[-6:]}"
    _store_roundtrip(
        tmp_path, corpus, {"provider": "pgvector", "connection": PG_URL, "collection_name": name}
    )


@pytest.mark.skipif(not _reachable(WEAVIATE_URL, 8080), reason="no Weaviate")
def test_weaviate(tmp_path, corpus):
    name = f"Rf_{os.getpid()}_{tmp_path.name[-6:]}"
    _store_roundtrip(
        tmp_path,
        corpus,
        {
            "provider": "weaviate",
            "url": WEAVIATE_URL,
            "grpc_port": WEAVIATE_GRPC,
            "collection_name": name,
        },
    )


def test_huggingface_embeddings_semantic_chunks_and_reranker(tmp_path, corpus):
    pytest.importorskip("sentence_transformers")
    config = RAGConfig.model_validate(
        {
            "persist_dir": str(tmp_path),
            "embeddings": HF_EMBEDDINGS,
            "vector_store": {"provider": "chroma"},
            "chunking": {"strategy": "semantic", "chunk_size": 400, "chunk_overlap": 0},
            "retrieval": {
                "top_k": 2,
                "hybrid": {"enabled": True},
                "rerank": {
                    "provider": "huggingface",
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                },
            },
        }
    )
    with RAGPipeline(config) as pipeline:
        pipeline.ingest(corpus)
        documents = pipeline.retrieve("How much of the sunlight do photovoltaic cells convert?")
    assert documents[0].metadata["source"].endswith("solar.md")
    assert "rerank_score" in documents[0].metadata


@pytest.mark.skipif(not _ollama_ready(), reason=f"Ollama model {OLLAMA_MODEL} not available")
def test_ollama_end_to_end(tmp_path, corpus):
    config = RAGConfig.model_validate(
        {
            "persist_dir": str(tmp_path),
            "embeddings": HF_EMBEDDINGS,
            "vector_store": {"provider": "chroma"},
            "chunking": {"strategy": "llm", "chunk_size": 600, "chunk_overlap": 0},
            "llm": _ollama_llm(),
            "retrieval": {
                "top_k": 2,
                "hybrid": {"enabled": True},
                "query_transform": {"strategies": ["multi_query"], "num_queries": 2},
                "corrective": {"enabled": True, "max_retries": 1},
            },
        }
    )
    with RAGPipeline(config) as pipeline:
        pipeline.ingest(corpus)
        result = pipeline.query("What is the typical efficiency of silicon solar cells?")
        streamed = "".join(pipeline.stream("What does a sourdough starter need?"))
        report = pipeline.evaluate(
            [
                EvalSample(
                    question="What is the efficiency of silicon solar cells?",
                    reference_answer="About 20 percent.",
                    relevant_sources=["solar.md"],
                )
            ],
            k_values=[1, 2],
        )
    print("\nANSWER:", result.answer, "\nQUERIES:", result.queries, "\nSTREAMED:", streamed)
    print("SUMMARY:", report.summary)
    assert "20" in result.answer
    assert len(result.queries) >= 2
    assert "flour" in streamed.lower() or "water" in streamed.lower()
    assert report.summary["hit_rate@1"]["mean"] == 1.0
    assert report.summary["faithfulness"]["count"] == 1


@pytest.mark.skipif(not _ollama_ready(), reason=f"Ollama model {OLLAMA_MODEL} not available")
def test_ragas_backend(tmp_path, corpus):
    from ragfoundry.evaluation.ragas_adapter import _import_ragas

    _import_ragas()
    config = RAGConfig.model_validate(
        {
            "persist_dir": str(tmp_path),
            "embeddings": HF_EMBEDDINGS,
            "vector_store": {"provider": "qdrant"},
            "llm": _ollama_llm(),
            "retrieval": {"top_k": 2},
        }
    )
    with RAGPipeline(config) as pipeline:
        pipeline.ingest(corpus)
        report = pipeline.evaluate(
            [
                EvalSample(
                    question="What is the efficiency of silicon solar cells?",
                    reference_answer="About 20 percent.",
                )
            ],
            mode="generation",
            backend="ragas",
        )
    print("\nRAGAS:", report.summary, report.rows[0].get("generation_metrics"))
    assert report.summary, "RAGAS returned no metrics"
    assert any(stats["count"] == 1 for stats in report.summary.values())


@pytest.mark.skipif(not _ollama_ready(), reason=f"Ollama model {OLLAMA_MODEL} not available")
def test_ollama_chat_follow_up_and_context_budget(tmp_path, corpus):
    config = RAGConfig.model_validate(
        {
            "persist_dir": str(tmp_path),
            "vector_store": {"provider": "chroma"},
            "llm": _ollama_llm(),
            "context_window": {"max_context_tokens": 120, "max_history_turns": 3},
        }
    )
    with RAGPipeline(config, embeddings=KeywordEmbeddings()) as rag:
        rag.ingest(corpus)
        chat = rag.chat()
        chat.ask("What do solar panels do?")
        follow_up = chat.ask("And how efficient are they?")
        assert follow_up.standalone_question is not None
        assert "solar" in follow_up.standalone_question.lower()
        assert follow_up.sources[0].endswith("solar.md")
        assert "20" in follow_up.answer
        assert 0 < follow_up.context_tokens <= 120
