# Claude

Anthropic's Claude models through the Claude API.

| | |
|---|---|
| Extra | `claude` |
| Key | `ANTHROPIC_API_KEY` ([get one](https://console.anthropic.com/)) |
| Models | [docs.anthropic.com/en/docs/about-claude/models](https://docs.anthropic.com/en/docs/about-claude/models) |

## Install

```bash
pip install "ragfoundry[claude]"
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Configure

=== "Python"

    ```python
    "llm": {"provider": "claude", "model": "claude-sonnet-5-5", "max_tokens": 1024}
    ```

=== "YAML"

    ```yaml
    llm:
      provider: claude
      model: claude-sonnet-5-5
      max_tokens: 1024
    ```

| Model | Notes |
|---|---|
| `claude-opus-5-5` | Most capable |
| `claude-sonnet-5-5` | Balanced quality and speed; a good default for RAG |
| `claude-haiku-5-5` | Fastest and cheapest; good for query rewriting and grading |

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
    "vector_store": {"provider": "chroma", "collection_name": "claude"},
    "llm": {"provider": "claude", "model": "claude-sonnet-5-5", "max_tokens": 1024},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```

## Notes

- `base_url` points at a proxy or gateway if you use one.
- `options` is passed to LangChain's `ChatAnthropic`.
- Anthropic doesn't offer embedding models; pair Claude with Hugging Face, OpenAI or Gemini
  embeddings.
