"""Smallest complete pipeline: ingest a folder and ask a question.

Runs fully on your machine with Hugging Face embeddings, Chroma and Ollama.

    pip install "ragfoundry[huggingface,chroma,ollama]"
    ollama pull llama3.2
    python examples/01_quickstart.py
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
        "vector_store": {"provider": "chroma", "collection_name": "quickstart"},
        "llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0},
    }
)

with RAGPipeline(config) as rag:
    report = rag.ingest(DOCS)
    print(f"Indexed {report.chunks_total} chunks ({report.chunks_added} new)")

    result = rag.query("How efficient are silicon solar cells?")
    print("Answer:", result.answer)
    print("Sources:", result.sources)
