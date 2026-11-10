"""Build a pipeline in code instead of YAML. Runs locally with Ollama and Hugging Face."""

from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate(
    {
        "embeddings": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "vector_store": {"provider": "chroma", "collection_name": "quickstart"},
        "llm": {"provider": "ollama", "model": "llama3.2"},
        "chunking": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 100},
        "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
    }
)

with RAGPipeline(config) as rag:
    print(rag.ingest("./docs").to_dict())
    result = rag.query("What are the main topics in these documents?")
    print(result.answer)
    print("Sources:", result.sources)

    report = rag.evaluate("examples/eval.jsonl", mode="retrieval")
    for metric, stats in report.summary.items():
        print(f"{metric:<16} {stats['mean']:.3f}")
