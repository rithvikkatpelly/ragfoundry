# Configure with Python or YAML

Every pipeline setting lives in one `RAGConfig`. You can build it from a Python dict or load it
from a YAML file; both accept exactly the same settings.

## From Python

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "kb"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "chunking": {"chunk_size": 800, "chunk_overlap": 150},
    "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
})
rag = RAGPipeline(config)
```

## From YAML

```yaml title="rag.yaml"
embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2
vector_store:
  provider: chroma
  collection_name: kb
llm:
  provider: groq
  model: openai/gpt-oss-120b
chunking:
  chunk_size: 800
  chunk_overlap: 150
retrieval:
  top_k: 4
  hybrid:
    enabled: true
```

```python
rag = RAGPipeline.from_yaml("rag.yaml")
# or: config = RAGConfig.from_yaml("rag.yaml")
```

YAML is handy when non-programmers tune settings, or when you keep one file per environment
(`dev.yaml`, `prod.yaml`).

## Environment variables in YAML

`${NAME}` is replaced with the environment variable `NAME`, and `${NAME:-default}` uses `default`
when it isn't set:

```yaml
llm:
  provider: groq
  model: ${GROQ_MODEL:-openai/gpt-oss-120b}
  api_key: ${GROQ_API_KEY}
vector_store:
  provider: qdrant
  url: ${QDRANT_URL:-http://localhost:6333}
```

A variable that isn't set and has no default stops with
`Environment variable 'X' is referenced but not set`, rather than silently becoming empty.

## What's required

| Section | Required for |
|---|---|
| `embeddings` | Ingesting and searching |
| `vector_store` | Ingesting and searching |
| `llm` | Answers, chat, query transforms, corrective retrieval, LLM chunking, evaluation |

Everything else has defaults. Plain retrieval (`rag.retrieve()`) works without an `llm`.

## Typos are caught

Unknown keys are rejected immediately:

```text
ValidationError: llm.temprature
  Extra inputs are not permitted
```

Values are checked too: `chunk_overlap` must be smaller than `chunk_size`, `fetch_k` at least
`top_k`, and so on.

## Inspecting and changing a config

`RAGConfig` is a [Pydantic](https://docs.pydantic.dev) model:

```python
print(config.retrieval.top_k)
print(config.model_dump())                          # everything as a dict, with defaults filled in
bigger = config.model_copy(update={"retrieval": config.retrieval.model_copy(update={"top_k": 8})})
```

Secrets such as API keys print as `**********`.

## Where state is kept

`persist_dir` (default `.ragfoundry/`) holds the chunk record used for incremental ingestion and
hybrid search, plus the data of local vector stores (Chroma, FAISS, Qdrant local, Milvus Lite).
Use a different `persist_dir` or `collection_name` for each separate dataset.

Every setting is listed in the [configuration reference](../reference/configuration.md).
