---
hide:
  - navigation
  - toc
---

# ragfoundry

**Build a production RAG system in Python with a single configuration.**

ragfoundry lets you chat with your own documents using any large language model. You choose the
pieces (where your documents come from, how they are split, which embedding model, vector
database and LLM to use) and ragfoundry connects them into a working, tested pipeline built on
LangChain and LangGraph.

```python
from ragfoundry import RAGConfig, RAGPipeline

rag = RAGPipeline(RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
}))

rag.ingest("./my_documents")                        # read, split and index your files
print(rag.query("What is our refund policy?").answer)
```

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **New to RAG?**

    ---

    Learn what RAG is and build your first pipeline in about 10 minutes, step by step.

    [:octicons-arrow-right-24: Get started](get-started/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Complete projects: chat with a PDF, a Streamlit app, a local RAG with no API keys, and a
    chatbot API.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-puzzle:{ .lg .middle } **Integrations**

    ---

    One page for every loader, chunker, embedding model, vector store, LLM, reranker and tracing
    tool, with setup and examples.

    [:octicons-arrow-right-24: Integrations](integrations/index.md)

-   :material-tools:{ .lg .middle } **How-to guides**

    ---

    Chat history, context window limits, hybrid search, reranking, evaluation, production
    deployment and more.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

</div>

## What's included

| Stage | Options |
|---|---|
| [Document loaders](integrations/loaders/index.md) | PDF, Word, TXT, Markdown, CSV, Excel, web pages and site crawling |
| [Chunking](integrations/chunking/index.md) | Fixed, recursive, semantic, LLM-based; sizes in characters or tokens |
| [Embeddings](integrations/embeddings/index.md) | Hugging Face (free, local), OpenAI, Gemini |
| [Vector stores](integrations/vector-stores/index.md) | Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, PGVector |
| [LLMs](integrations/llms/index.md) | Groq, OpenAI, Gemini, Claude, Ollama (local) |
| [Retrieval](how-to/hybrid-search.md) | Hybrid keyword + vector search, reranking, query rewriting, corrective retrieval, metadata filters |
| [Chat](how-to/chat-history.md) | Conversations with memory and follow-up question rewriting |
| [Context window](how-to/context-window.md) | Token budgets for retrieved text and chat history |
| [Evaluation](how-to/evaluation.md) | Retrieval and answer-quality metrics, optional RAGAS |
| [Tracing](how-to/tracing.md) | LangSmith or Langfuse |

## Install

```bash
pip install "ragfoundry[pdf,huggingface,chroma,groq]"
```

Then follow [Your first RAG pipeline](get-started/first-pipeline.md).
