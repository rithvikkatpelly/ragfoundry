# Trace with LangSmith or Langfuse

Tracing records every step of every request (the search queries, retrieved passages, prompts,
LLM replies, timings and token use) in a web dashboard. It's the fastest way to see *why* an
answer was wrong, and to watch cost and latency in production.

=== "LangSmith"

    ```bash
    pip install "ragfoundry[langsmith]"
    ```

    ```python
    "observability": {
        "provider": "langsmith",
        "api_key": "lsv2_...",            # or ${LANGSMITH_API_KEY} in YAML
        "project": "my-rag",
    }
    ```

    Details: [LangSmith integration](../integrations/observability/langsmith.md)

=== "Langfuse"

    ```bash
    pip install "ragfoundry[langfuse]"
    ```

    ```python
    "observability": {
        "provider": "langfuse",
        "public_key": "pk-lf-...",
        "secret_key": "sk-lf-...",
        "endpoint": "https://cloud.langfuse.com",   # or your self-hosted URL
    }
    ```

    Details: [Langfuse integration](../integrations/observability/langfuse.md)

## What you'll see

Each call is one trace: `ragfoundry.ingest`, `ragfoundry.query`, `ragfoundry.retrieve`,
`ragfoundry.stream` and the evaluation runs. Inside each are the graph steps (`condense`,
`transform`, `retrieve`, `rerank`, `grade`, `generate`) and every LLM call with its prompt and
reply.

## Tags

```python
"observability": {"provider": "langsmith", "api_key": "...", "tags": ["prod", "v2-prompt"]}
```

Tags are attached to every trace, so you can filter by environment or experiment.

## Flushing

Traces are sent in the background. Close the pipeline before your program exits so none are lost:

```python
with RAGPipeline(config) as rag:      # closes automatically
    rag.query("...")

rag.flush()                           # or send buffered traces now
rag.close()                           # flushes and closes connections
```
