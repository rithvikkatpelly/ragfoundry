# Gemini embeddings

Google's hosted embedding models through the Gemini API.

| | |
|---|---|
| Extra | `gemini` |
| Key | `GOOGLE_API_KEY` ([get one](https://aistudio.google.com/apikey)) |
| Pricing | Free tier with rate limits, then per token |

## Install

```bash
pip install "ragfoundry[gemini]"
export GOOGLE_API_KEY="..."
```

## Configure

=== "Python"

    ```python
    "embeddings": {"provider": "gemini", "model": "models/gemini-embedding-001"}
    ```

=== "YAML"

    ```yaml
    embeddings:
      provider: gemini
      model: models/gemini-embedding-001
    ```

`options` is passed to LangChain's `GoogleGenerativeAIEmbeddings`.

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "gemini", "model": "models/gemini-embedding-001"},
    "vector_store": {"provider": "chroma", "collection_name": "gemini_docs"},
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash"},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```

## Notes

- The free tier limits requests per minute. Ingesting a large folder may hit the limit; lower
  `batch_size` or ingest in parts.
- Check [Google's model list](https://ai.google.dev/gemini-api/docs/models) for current embedding
  models.
