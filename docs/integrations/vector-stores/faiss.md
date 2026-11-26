# FAISS

[FAISS](https://github.com/facebookresearch/faiss) is Meta's library for fast vector search. It
runs inside your process and ragfoundry saves the index to files.

| | |
|---|---|
| Extra | `faiss` |
| Setup | None |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[faiss]"
```

## Configure

```python
"vector_store": {"provider": "faiss", "collection_name": "my_docs"}
```

The index is saved to `persist_dir/faiss/<collection_name>` after each ingest. Choose a folder with
`path`:

```python
"vector_store": {"provider": "faiss", "path": "./indexes/handbook"}
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `collection_name` | `ragfoundry` | Names the index folder |
| `path` | `persist_dir/faiss/<collection_name>` | Where the index files are saved |

## Notes

!!! warning "macOS with Hugging Face models"
    The `faiss-cpu` and PyTorch packages each include their own copy of OpenMP, and loading both
    in one process on macOS can crash it with `OMP: Error #15`. ragfoundry warns when it sees this
    combination. On macOS, use [Chroma](chroma.md) or [Qdrant](qdrant.md) with Hugging Face
    embeddings or rerankers, or use hosted embeddings with FAISS.

- The whole index is loaded into memory.
- One process should write to an index at a time.
- Filtering happens after the nearest-neighbour search, so very selective filters may return fewer
  than `top_k` results.
