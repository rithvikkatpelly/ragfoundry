# OpenAI

OpenAI's GPT models through the OpenAI API.

| | |
|---|---|
| Extra | `openai` |
| Key | `OPENAI_API_KEY` ([get one](https://platform.openai.com/api-keys)) |
| Models | [platform.openai.com/docs/models](https://platform.openai.com/docs/models) |

## Install

```bash
pip install "ragfoundry[openai]"
export OPENAI_API_KEY="sk-..."
```

## Configure

=== "Python"

    ```python
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0}
    ```

=== "YAML"

    ```yaml
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0
      api_key: ${OPENAI_API_KEY}     # optional
    ```

Reasoning models may reject `temperature`; leave it out for those.

## OpenAI-compatible servers

Many servers implement the OpenAI API: vLLM, LM Studio, llama.cpp server, Together, OpenRouter,
company gateways. Point `base_url` at them:

```python
"llm": {
    "provider": "openai",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
}
```

For Azure OpenAI, pass LangChain's `AzureChatOpenAI` directly; see
[Custom components](../../how-to/custom-components.md).

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
    "vector_store": {"provider": "chroma", "collection_name": "openai"},
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```

`options` is passed to LangChain's `ChatOpenAI`.
