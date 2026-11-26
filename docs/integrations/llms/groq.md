# Groq

[Groq](https://groq.com) runs open models such as Llama, Qwen and GPT-OSS on its own hardware,
with very fast responses and a free tier. A good first LLM for learning and prototypes.

| | |
|---|---|
| Extra | `groq` |
| Key | `GROQ_API_KEY` ([get one](https://console.groq.com/keys)) |
| Models | [console.groq.com/docs/models](https://console.groq.com/docs/models) |

## Install

```bash
pip install "ragfoundry[groq]"
export GROQ_API_KEY="gsk_..."
```

## Configure

=== "Python"

    ```python
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0}
    ```

=== "YAML"

    ```yaml
    llm:
      provider: groq
      model: ${GROQ_MODEL:-openai/gpt-oss-120b}
      temperature: 0
    ```

=== "Colab"

    ```python
    from google.colab import userdata
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b",
            "api_key": userdata.get("GROQ_API_KEY")}
    ```

## Models

Examples at the time of writing; check Groq's list for what's current.

| Model | Notes |
|---|---|
| `openai/gpt-oss-120b` | Strong general model |
| `openai/gpt-oss-20b` | Faster and cheaper |
| `llama-3.3-70b-versatile` | Llama 3.3 70B |
| `llama-3.1-8b-instant` | Very fast, for query rewriting or grading |

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0, "max_tokens": 800},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    for token in rag.stream("Summarise our refund policy"):
        print(token, end="", flush=True)
```

## Notes

- The free tier has per-minute and per-day rate limits. Corrective retrieval and evaluation make
  many calls and may hit them; lower `evaluation.max_concurrency`, or upgrade.
- `options` is passed to LangChain's `ChatGroq`.
