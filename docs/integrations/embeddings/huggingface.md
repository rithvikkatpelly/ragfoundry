# Hugging Face embeddings

Runs open-source [sentence-transformers](https://www.sbert.net) models on your own machine. Free,
private and works offline after the first download.

| | |
|---|---|
| Extra | `huggingface` |
| Key | none (`HF_TOKEN` only for gated models) |
| Runs | CPU, NVIDIA GPU (CUDA) or Apple Silicon (MPS) |

## Install

```bash
pip install "ragfoundry[huggingface]"
```

This installs PyTorch, which is a large download the first time.

## Configure

=== "Python"

    ```python
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    ```

=== "YAML"

    ```yaml
    embeddings:
      provider: huggingface
      model: sentence-transformers/all-MiniLM-L6-v2
    ```

The model downloads from the Hugging Face Hub on first use and is cached in
`~/.cache/huggingface`.

## Options

`options` is passed to LangChain's `HuggingFaceEmbeddings`:

```python
"embeddings": {
    "provider": "huggingface",
    "model": "BAAI/bge-small-en-v1.5",
    "options": {
        "model_kwargs": {"device": "cuda"},                 # or "mps" on Apple Silicon, "cpu"
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 64},
    },
}
```

## Recommended models

| Model | Size | Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 90 MB | Fast default for English |
| `BAAI/bge-small-en-v1.5` | 130 MB | Better English retrieval |
| `BAAI/bge-base-en-v1.5` | 440 MB | Higher quality, slower |
| `intfloat/multilingual-e5-base` | 1.1 GB | 100 languages |

Browse more on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
    "vector_store": {"provider": "chroma", "collection_name": "bge_docs"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("What is the refund policy?").answer)
```

## Notes

- Ingesting many documents on a CPU can be slow; a GPU is 10 to 50 times faster.
- On macOS, don't combine Hugging Face models with [FAISS](../vector-stores/faiss.md) in one
  process. Use Chroma or Qdrant.
