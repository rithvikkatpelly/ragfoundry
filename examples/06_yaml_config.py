"""Load the pipeline from rag.yaml instead of building the configuration in code.

    pip install "ragfoundry[huggingface,chroma,groq]"
    export GROQ_API_KEY="gsk_..."
    python examples/06_yaml_config.py
"""

from pathlib import Path

from ragfoundry import RAGPipeline

HERE = Path(__file__).parent

with RAGPipeline.from_yaml(HERE / "rag.yaml") as rag:
    rag.ingest(HERE / "sample_docs")
    print("Indexed:", rag.sources())

    chat = rag.chat()
    print(chat.ask("What does a sourdough starter need?").answer)
    follow_up = chat.ask("How often?")
    print("Searched for:", follow_up.standalone_question)
    print(follow_up.answer)

    # Remove a source from the index.
    removed = rag.delete_source(HERE / "sample_docs" / "parts.csv")
    print(f"Removed {removed} chunk(s)")
