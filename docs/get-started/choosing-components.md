# Choosing components

Not sure which provider to pick? Start with the recommended setup for your situation, get it
working, then [evaluate](../how-to/evaluation.md) before changing anything.

## Recommended setups

=== "Getting started (free)"

    Free local embeddings, a local vector store and Groq's free API.

    ```bash
    pip install "ragfoundry[loaders,huggingface,chroma,groq]"
    ```

    ```python
    {
        "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
        "vector_store": {"provider": "chroma"},
        "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    }
    ```

=== "Private / offline"

    Nothing leaves your computer.

    ```bash
    pip install "ragfoundry[loaders,huggingface,chroma,ollama]"
    ollama pull llama3.2
    ```

    ```python
    {
        "embeddings": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
        "vector_store": {"provider": "chroma"},
        "llm": {"provider": "ollama", "model": "llama3.2"},
    }
    ```

=== "Production (hosted)"

    Managed services that scale, with tracing.

    ```bash
    pip install "ragfoundry[loaders,openai,qdrant,huggingface,langfuse]"
    ```

    ```python
    {
        "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"provider": "qdrant", "url": "https://YOUR-CLUSTER.qdrant.io",
                         "api_key": "..."},
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "retrieval": {"hybrid": {"enabled": True},
                      "rerank": {"provider": "huggingface",
                                 "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}},
        "observability": {"provider": "langfuse", "public_key": "...", "secret_key": "..."},
    }
    ```

=== "Already using Postgres"

    Keep vectors next to your application data.

    ```bash
    pip install "ragfoundry[loaders,openai,pgvector]"
    ```

    ```python
    {
        "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"provider": "pgvector",
                         "connection": "postgresql+psycopg://user:pass@host:5432/db"},
        "llm": {"provider": "claude", "model": "claude-sonnet-5-5"},
    }
    ```

## Embedding model

| Choose | When |
|---|---|
| [Hugging Face](../integrations/embeddings/huggingface.md) | Free, private, no key. Good quality for English. Uses your CPU/GPU at ingest time. |
| [OpenAI](../integrations/embeddings/openai.md) | Strong quality across languages, nothing to run locally, small cost per document. |
| [Gemini](../integrations/embeddings/gemini.md) | You already use Google AI; generous free tier. |

The embedding model is the hardest part to change later, because changing it means re-indexing
everything.

## Vector store

| Choose | When |
|---|---|
| [Chroma](../integrations/vector-stores/chroma.md) | Default. Local folder, nothing to install. Fine up to around a million chunks. |
| [FAISS](../integrations/vector-stores/faiss.md) | Fast in-process search saved to files. Avoid on macOS with Hugging Face models. |
| [Qdrant](../integrations/vector-stores/qdrant.md) | Local mode for development, then the same code against a Qdrant server or cloud. |
| [PGVector](../integrations/vector-stores/pgvector.md) | You already run PostgreSQL. |
| [Pinecone](../integrations/vector-stores/pinecone.md) | Fully managed and serverless; no servers to run. |
| [Weaviate](../integrations/vector-stores/weaviate.md) | You use Weaviate already, self-hosted or cloud. |
| [Milvus](../integrations/vector-stores/milvus.md) | Very large collections; Milvus Lite for local development. |

## LLM

| Choose | When |
|---|---|
| [Groq](../integrations/llms/groq.md) | Free tier, very fast open models. Great for learning and prototypes. |
| [OpenAI](../integrations/llms/openai.md) | Reliable, high quality, wide model range. |
| [Claude](../integrations/llms/claude.md) | Careful, well-cited answers over long contexts. |
| [Gemini](../integrations/llms/gemini.md) | Long context windows, free tier. |
| [Ollama](../integrations/llms/ollama.md) | Private, offline, free; needs a decent computer. |

## Chunking

Start with `recursive`, `chunk_size: 1000`, `chunk_overlap: 200`. See
[Choose chunk size](../how-to/chunk-size.md) for when to change it.

## Retrieval features

Turn these on one at a time and measure each:

1. [Hybrid search](../how-to/hybrid-search.md): almost always helps; costs very little.
2. [Reranking](../how-to/reranking.md): a clear gain for most document sets; adds 100 to 500 ms.
3. [Query transforms](../how-to/query-transforms.md): helps vague or short questions; adds an LLM call.
4. [Corrective retrieval](../how-to/corrective-retrieval.md): helps when wrong passages are often
   retrieved; adds several LLM calls.
