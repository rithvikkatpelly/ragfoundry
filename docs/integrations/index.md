# Integrations

ragfoundry connects to these providers out of the box. Each has its own page with installation,
credentials, configuration options and a complete example.

## Document loaders

| Source | Extensions | Extra |
|---|---|---|
| [PDF](loaders/pdf.md) | `.pdf` | `pdf` |
| [Word](loaders/docx.md) | `.docx` | `docx` |
| [Text and Markdown](loaders/text.md) | `.txt`, `.md`, `.markdown` | none |
| [CSV](loaders/csv.md) | `.csv` | none |
| [Excel](loaders/excel.md) | `.xlsx`, `.xlsm`, `.xls` | `excel` |
| [Web pages](loaders/web.md) | `http://`, `https://` | `web` |

## Chunking strategies

| Strategy | Best for | Extra cost |
|---|---|---|
| [Recursive](chunking/recursive.md) (default) | Almost everything | None |
| [Fixed](chunking/fixed.md) | Uniform text without structure | None |
| [Semantic](chunking/semantic.md) | Long documents that change topic | Embedding calls at ingest |
| [LLM](chunking/llm.md) | Highest-quality topic boundaries | LLM calls at ingest |

## Embedding models

| Provider | Runs | Key | Extra |
|---|---|---|---|
| [Hugging Face](embeddings/huggingface.md) | Locally, free | None | `huggingface` |
| [OpenAI](embeddings/openai.md) | Hosted | `OPENAI_API_KEY` | `openai` |
| [Gemini](embeddings/gemini.md) | Hosted | `GOOGLE_API_KEY` | `gemini` |

## Vector stores

| Store | Runs | Extra |
|---|---|---|
| [Chroma](vector-stores/chroma.md) | Local folder or server | `chroma` |
| [FAISS](vector-stores/faiss.md) | Local files | `faiss` |
| [Pinecone](vector-stores/pinecone.md) | Managed cloud | `pinecone` |
| [Qdrant](vector-stores/qdrant.md) | Local folder, server or cloud | `qdrant` |
| [Weaviate](vector-stores/weaviate.md) | Server or cloud | `weaviate` |
| [Milvus](vector-stores/milvus.md) | Local file (Milvus Lite), server or cloud | `milvus` |
| [PGVector](vector-stores/pgvector.md) | PostgreSQL | `pgvector` |

## Chat models (LLMs)

| Provider | Runs | Key | Extra |
|---|---|---|---|
| [Groq](llms/groq.md) | Hosted, free tier | `GROQ_API_KEY` | `groq` |
| [OpenAI](llms/openai.md) | Hosted | `OPENAI_API_KEY` | `openai` |
| [Gemini](llms/gemini.md) | Hosted, free tier | `GOOGLE_API_KEY` | `gemini` |
| [Claude](llms/claude.md) | Hosted | `ANTHROPIC_API_KEY` | `claude` |
| [Ollama](llms/ollama.md) | Locally, free | None | `ollama` |

## Rerankers

| Provider | Runs | Extra |
|---|---|---|
| [Hugging Face cross-encoder](rerankers/huggingface.md) | Locally, free | `huggingface` |
| [Cohere](rerankers/cohere.md) | Hosted | `cohere` |

## Observability

| Provider | Extra |
|---|---|
| [LangSmith](observability/langsmith.md) | `langsmith` |
| [Langfuse](observability/langfuse.md) | `langfuse` |

## Something else?

Any LangChain chat model, embedding model or vector store can be passed in directly, and you can
register loaders for new file types. See [Custom components](../how-to/custom-components.md).
