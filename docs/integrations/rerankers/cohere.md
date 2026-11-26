# Cohere Rerank

Cohere's hosted [Rerank](https://cohere.com/rerank) API: strong, multilingual, nothing to run
locally.

| | |
|---|---|
| Extra | `cohere` |
| Key | `COHERE_API_KEY` ([get one](https://dashboard.cohere.com/api-keys)) |

## Install

```bash
pip install "ragfoundry[cohere]"
export COHERE_API_KEY="..."
```

## Configure

=== "Python"

    ```python
    "retrieval": {
        "top_k": 5,
        "fetch_k": 30,
        "rerank": {"provider": "cohere", "model": "rerank-v3.5"},
    }
    ```

=== "YAML"

    ```yaml
    retrieval:
      top_k: 5
      fetch_k: 30
      rerank:
        provider: cohere
        model: rerank-v3.5
        api_key: ${COHERE_API_KEY}
    ```

Check [Cohere's model list](https://docs.cohere.com/docs/rerank) for current model names.

## Notes

- Each question is one API call, billed per search.
- Trial keys are rate limited.
