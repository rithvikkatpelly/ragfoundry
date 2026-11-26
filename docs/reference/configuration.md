# Configuration reference

All settings with their defaults. Use them as a Python dict with `RAGConfig.model_validate()` or in
a YAML file with `RAGPipeline.from_yaml()`; see [Configure with Python or YAML](../how-to/configuration.md).

Unknown keys are rejected, so typos fail immediately.

```yaml
persist_dir: .ragfoundry       # local state: chunk record, and Chroma/FAISS/Qdrant/Milvus Lite data
batch_size: 64                 # chunks written to the vector store per batch

loading:
  on_error: raise              # raise | skip
  encoding: utf-8
  glob: "**/*"                 # which files to take from a folder
  web:
    max_depth: 0               # 0 fetches only the given URL
    max_pages: 50
    same_domain: true
    respect_robots: true
    delay_seconds: 0.5
    timeout_seconds: 30
    max_bytes: 10000000
    user_agent: ragfoundry/0.2

chunking:
  strategy: recursive          # fixed | recursive | semantic | llm | a registered name
  chunk_size: 1000
  chunk_overlap: 200
  length_unit: characters      # characters | tokens
  encoding_name: cl100k_base   # tokenizer used when length_unit is tokens
  separators: null             # custom separators for recursive
  semantic:
    breakpoint_percentile: 95
    buffer_size: 1
  llm:
    window_chars: 8000

embeddings:                    # required for ingest and search
  provider: huggingface        # huggingface | openai | gemini
  model: sentence-transformers/all-MiniLM-L6-v2
  api_key: null                # default: the provider's environment variable
  base_url: null               # OpenAI-compatible endpoints
  options: {}                  # passed to the LangChain embeddings class

vector_store:                  # required; one provider block
  provider: chroma
  collection_name: ragfoundry
  # chroma:   persist_directory, or host and port
  # faiss:    path
  # pinecone: index_name (required), api_key; collection_name is used as the namespace
  # qdrant:   url and api_key, or path for local mode
  # weaviate: url, grpc_port, api_key (cloud)
  # milvus:   uri and token; omit uri for Milvus Lite
  # pgvector: connection (required)

llm:                           # required for answers
  provider: ollama             # groq | openai | gemini | claude | ollama
  model: llama3.2
  api_key: null
  base_url: null
  temperature: null            # sent only when set
  max_tokens: null
  options: {}                  # passed to the LangChain chat model class

retrieval:
  top_k: 5
  fetch_k: 20                  # must be >= top_k
  hybrid: {enabled: false, rrf_k: 60, vector_weight: 1.0, bm25_weight: 1.0}
  rerank: null                 # {provider: huggingface | cohere, model, api_key}
  query_transform: {strategies: [], num_queries: 3}    # rewrite | multi_query | hyde
  corrective: {enabled: false, max_retries: 1, min_relevant_docs: 1}

context_window:
  max_context_tokens: null     # token budget for retrieved passages; null = no limit
  truncate_last_passage: true
  max_history_turns: 5         # 0 disables chat history
  max_history_tokens: null
  condense_question: true
  token_counter: approximate   # approximate | tiktoken
  encoding_name: cl100k_base

generation:
  system_prompt: null          # replaces the default prompt
  abstain_message: "I don't have enough information in the provided documents to answer."

observability:
  provider: none               # none | langsmith | langfuse
  project: ragfoundry
  api_key: null                # LangSmith
  public_key: null             # Langfuse
  secret_key: null             # Langfuse
  endpoint: null
  tags: []

evaluation:
  llm: null                    # judge model; defaults to the pipeline llm
  k_values: [1, 3, 5]
  max_concurrency: 1
```

## Sections

| Section | Required | Guide |
|---|---|---|
| `persist_dir`, `batch_size` | | [Keep the index up to date](../how-to/managing-the-index.md) |
| `loading` | | [Load documents](../how-to/loading-documents.md), [Web pages](../integrations/loaders/web.md) |
| `chunking` | | [Chunking strategies](../integrations/chunking/index.md), [Choose chunk size](../how-to/chunk-size.md) |
| `embeddings` | Yes | [Embedding models](../integrations/embeddings/index.md) |
| `vector_store` | Yes | [Vector stores](../integrations/vector-stores/index.md) |
| `llm` | For answers | [Chat models](../integrations/llms/index.md) |
| `retrieval` | | [Hybrid search](../how-to/hybrid-search.md), [Reranking](../how-to/reranking.md), [Query transforms](../how-to/query-transforms.md), [Corrective retrieval](../how-to/corrective-retrieval.md) |
| `context_window` | | [Limit the context window](../how-to/context-window.md), [Chat with history](../how-to/chat-history.md) |
| `generation` | | [Customise the prompt](../how-to/prompts.md) |
| `observability` | | [Tracing](../how-to/tracing.md) |
| `evaluation` | | [Evaluate quality](../how-to/evaluation.md) |

## Environment variables in YAML

`${NAME}` reads an environment variable; `${NAME:-default}` falls back to `default`. An unset
variable without a default is an error.
