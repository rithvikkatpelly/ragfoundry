"""Hybrid search, reranking, multi-query, the corrective loop and a context budget.

    pip install "ragfoundry[huggingface,chroma,ollama]"
    python examples/03_advanced_retrieval.py
"""

from pathlib import Path

from ragfoundry import RAGConfig, RAGPipeline

DOCS = Path(__file__).parent / "sample_docs"

config = RAGConfig.model_validate(
    {
        "embeddings": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "vector_store": {"provider": "chroma", "collection_name": "advanced"},
        "llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0},
        "chunking": {"strategy": "recursive", "chunk_size": 400, "chunk_overlap": 50},
        "retrieval": {
            "top_k": 3,
            "fetch_k": 15,
            "hybrid": {"enabled": True},  # BM25 finds exact codes like "ZT-9902"
            "rerank": {
                "provider": "huggingface",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "query_transform": {"strategies": ["multi_query"], "num_queries": 3},
            "corrective": {"enabled": True, "max_retries": 1},
        },
        "context_window": {"max_context_tokens": 600},
    }
)

with RAGPipeline(config) as rag:
    rag.ingest(DOCS)

    result = rag.query("What does part ZT-9902 cost?")
    print("Answer:", result.answer)
    print("Queries searched:", result.queries)
    print("Corrective retries:", result.retries)
    print("Context tokens sent:", result.context_tokens)
    for number, doc in enumerate(result.documents, start=1):
        print(f"[{number}] {doc.metadata['source']}: {doc.page_content[:70]!r}")

    # Retrieval only (no answer generated), limited to CSV files.
    for doc in rag.retrieve("brake parts", where={"file_type": "csv"}):
        print("CSV hit:", doc.page_content[:60])
