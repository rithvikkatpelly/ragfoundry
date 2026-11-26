# Ollama

[Ollama](https://ollama.com) runs open models (Llama, Qwen, Mistral, Gemma...) on your own
computer. Free, private, and works offline.

| | |
|---|---|
| Extra | `ollama` |
| Key | none |
| Models | [ollama.com/library](https://ollama.com/library) |

## Install

1. Install Ollama from [ollama.com](https://ollama.com) and start it.
2. Download a model:

    ```bash
    ollama pull llama3.2
    ```

3. Install the extra:

    ```bash
    pip install "ragfoundry[ollama]"
    ```

## Configure

=== "Python"

    ```python
    "llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0}
    ```

=== "YAML"

    ```yaml
    llm:
      provider: ollama
      model: llama3.2
      temperature: 0
    ```

## Options

| Setting | Meaning |
|---|---|
| `base_url` | Ollama server, default `http://localhost:11434` |
| `max_tokens` | Maximum answer length (Ollama's `num_predict`) |
| `options` | Passed to LangChain's `ChatOllama`, for example `{"num_ctx": 8192}` or `{"reasoning": False}` |

## Models

| Model | Download | Needs |
|---|---|---|
| `llama3.2` | 2 GB | Most laptops |
| `qwen3:8b` | 5 GB | 16 GB RAM; add `"options": {"reasoning": False}` for quick answers |
| `llama3.1:8b` | 5 GB | 16 GB RAM |
| `gemma3:12b` | 8 GB | 24 GB RAM or a GPU |

## Context length

Ollama uses a small context window by default. If you send many passages or long chat history,
raise it and set a matching [context budget](../../how-to/context-window.md):

```python
"llm": {"provider": "ollama", "model": "llama3.2", "options": {"num_ctx": 8192}},
"context_window": {"max_context_tokens": 4000, "max_history_tokens": 1500},
```

## Complete example

See the [fully local RAG tutorial](../../tutorials/local-rag.md).

## Notes

- ragfoundry doesn't use Ollama for embeddings; for a fully local setup use
  [Hugging Face embeddings](../embeddings/huggingface.md).
- First answers are slow while the model loads into memory.
