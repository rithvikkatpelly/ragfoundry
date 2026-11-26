# Hugging Face cross-encoder reranker

Runs a [cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
model locally: it reads the question and each passage together and scores the match.

| | |
|---|---|
| Extra | `huggingface` |
| Key | none |

## Install

```bash
pip install "ragfoundry[huggingface]"
```

## Configure

```python
"retrieval": {
    "top_k": 5,
    "fetch_k": 20,
    "rerank": {"provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
}
```

## Models

| Model | Size | Notes |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90 MB | Fast; good English default |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 130 MB | Slightly better, slower |
| `BAAI/bge-reranker-base` | 1.1 GB | Strong, multilingual |
| `BAAI/bge-reranker-v2-m3` | 2.3 GB | Strongest multilingual; best with a GPU |

## Speed

Each candidate is one model pass: roughly 5 to 20 ms on a CPU with the MiniLM models, so
`fetch_k: 20` adds 100 to 400 ms per question. A GPU makes it much faster.

## Notes

- The model downloads on first use, like Hugging Face embedding models.
- On macOS, don't combine it with [FAISS](../vector-stores/faiss.md) in one process.
