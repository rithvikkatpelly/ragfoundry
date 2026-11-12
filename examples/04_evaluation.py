"""Measure retrieval and answer quality against a labelled question set.

    pip install "ragfoundry[huggingface,chroma,ollama]"
    python examples/04_evaluation.py
"""

from pathlib import Path

from ragfoundry import EvalSample, RAGConfig, RAGPipeline

HERE = Path(__file__).parent

config = RAGConfig.model_validate(
    {
        "embeddings": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "vector_store": {"provider": "chroma", "collection_name": "evaluation"},
        "llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0},
        "retrieval": {"top_k": 3, "hybrid": {"enabled": True}},
        "evaluation": {"k_values": [1, 3]},
    }
)

with RAGPipeline(config) as rag:
    rag.ingest(HERE / "sample_docs")

    # 1. Retrieval metrics only: fast, no LLM calls.
    report = rag.evaluate(HERE / "eval.jsonl", mode="retrieval")
    print("Retrieval:")
    for metric, stats in sorted(report.summary.items()):
        print(f"  {metric:<16} {stats['mean']:.3f}  (n={stats['count']})")

    # 2. Retrieval and generation, from samples built in code.
    samples = [
        EvalSample(
            question="How hot should the oven be for sourdough?",
            reference_answer="About 250 degrees Celsius.",
            relevant_sources=["sourdough.md"],
        ),
    ]
    full = rag.evaluate(samples, mode="all")
    print("All metrics:")
    for metric, stats in sorted(full.summary.items()):
        print(f"  {metric:<20} {stats['mean']:.3f}")
    full.save(HERE / "eval_report.json")  # per-question rows and any metric errors
