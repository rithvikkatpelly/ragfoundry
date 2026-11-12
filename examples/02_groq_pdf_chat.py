"""Chat with a PDF using Groq: follow-up questions remember the conversation.

    pip install "ragfoundry[huggingface,chroma,groq,pdf]"
    export GROQ_API_KEY="gsk_..."
    python examples/02_groq_pdf_chat.py path/to/file.pdf
"""

import os
import sys

from ragfoundry import RAGConfig, RAGPipeline

if len(sys.argv) != 2:
    sys.exit("Usage: python examples/02_groq_pdf_chat.py path/to/file.pdf")

config = RAGConfig.model_validate(
    {
        "embeddings": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "vector_store": {"provider": "chroma", "collection_name": "pdf_chat"},
        "llm": {
            "provider": "groq",
            "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "temperature": 0,
        },
        "chunking": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 150},
        "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
        "context_window": {
            "max_context_tokens": 3000,  # retrieved passages sent per question
            "max_history_turns": 5,  # earlier questions and answers sent per question
        },
    }
)

with RAGPipeline(config) as rag:
    print("Ingest:", rag.ingest(sys.argv[1]).to_dict())
    chat = rag.chat()
    print("Ask about the PDF. Follow-up questions work. Press Enter on an empty line to quit.")
    while question := input("\nYou: ").strip():
        print("Assistant: ", end="", flush=True)
        for token in chat.stream(question):
            print(token, end="", flush=True)
        print()
