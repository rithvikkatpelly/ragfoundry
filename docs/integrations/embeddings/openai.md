# OpenAI embeddings

OpenAI's hosted embedding models: strong quality, many languages, nothing to run locally.

| | |
|---|---|
| Extra | `openai` |
| Key | `OPENAI_API_KEY` ([get one](https://platform.openai.com/api-keys)) |
| Pricing | Per token embedded; see [openai.com/api/pricing](https://openai.com/api/pricing) |

## Install

```bash
pip install "ragfoundry[openai]"
export OPENAI_API_KEY="sk-..."
```

## Configure

=== "Python"

    ```python
    "embeddings": {"provider": "openai", "model": "text-embedding-3-small"}
    ```

=== "YAML"

    ```yaml
    embeddings:
      provider: openai
      model: text-embedding-3-small
      api_key: ${OPENAI_API_KEY}     # optional; read from the environment anyway
    ```

## Models

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Best value; a good default |
| `text-embedding-3-large` | 3072 | Highest quality |

Shorter vectors (cheaper storage) with the `dimensions` option:

```python
"embeddings": {"provider": "openai", "model": "text-embedding-3-large",
               "options": {"dimensions": 1024}}
```

## OpenAI-compatible servers

`base_url` points the client at any server that implements the OpenAI embeddings API, such as
vLLM, LM Studio, Together or a company gateway:

```python
"embeddings": {
    "provider": "openai",
    "model": "nomic-embed-text",
    "base_url": "http://localhost:1234/v1",
    "api_key": "not-needed",
    "options": {"check_embedding_ctx_length": False},
}
```

For Azure OpenAI, pass LangChain's `AzureOpenAIEmbeddings` directly; see
[Custom components](../../how-to/custom-components.md).

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
    "vector_store": {"provider": "chroma", "collection_name": "openai_docs"},
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```
