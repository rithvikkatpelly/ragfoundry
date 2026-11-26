# Rerankers

A reranker re-scores the candidates from first-stage retrieval and keeps the best `top_k`. It
usually gives a clear quality boost for a small latency cost. How and when to use one:
[Reranking](../../how-to/reranking.md).

| Provider | Runs | Key | Extra |
|---|---|---|---|
| [Hugging Face cross-encoder](huggingface.md) | Your machine | None | `huggingface` |
| [Cohere](cohere.md) | Hosted | `COHERE_API_KEY` | `cohere` |

```python
"retrieval": {
    "top_k": 5,
    "fetch_k": 20,
    "rerank": {"provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
}
```

| Setting | Meaning |
|---|---|
| `provider` | `huggingface` or `cohere` |
| `model` | Model name |
| `api_key` | Cohere only; defaults to `COHERE_API_KEY` |

Each reranked passage gets `metadata["rerank_score"]`.
