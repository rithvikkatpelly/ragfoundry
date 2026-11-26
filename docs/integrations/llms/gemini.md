# Gemini

Google's Gemini models through the Gemini API (Google AI Studio).

| | |
|---|---|
| Extra | `gemini` |
| Key | `GOOGLE_API_KEY` ([get one](https://aistudio.google.com/apikey)) |
| Models | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) |

## Install

```bash
pip install "ragfoundry[gemini]"
export GOOGLE_API_KEY="..."
```

## Configure

=== "Python"

    ```python
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0}
    ```

=== "YAML"

    ```yaml
    llm:
      provider: gemini
      model: gemini-2.5-flash
      temperature: 0
    ```

## Complete example

Gemini for both embeddings and answers, one key for everything:

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "gemini", "model": "models/gemini-embedding-001"},
    "vector_store": {"provider": "chroma", "collection_name": "gemini"},
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```

## Notes

- Gemini models have very large context windows, but sending more passages isn't always better;
  set a [context budget](../../how-to/context-window.md).
- The free tier is rate limited.
- `options` is passed to LangChain's `ChatGoogleGenerativeAI`.
