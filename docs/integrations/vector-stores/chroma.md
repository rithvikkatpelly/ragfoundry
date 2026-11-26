# Chroma

[Chroma](https://www.trychroma.com) is an open-source vector database that can run inside your
Python process, saving to a local folder, or as a separate server. It's the easiest store to start
with.

| | |
|---|---|
| Extra | `chroma` |
| Setup | None for local use |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[chroma]"
```

## Local (embedded)

```python
"vector_store": {"provider": "chroma", "collection_name": "my_docs"}
```

Data is saved in `persist_dir/chroma` (by default `.ragfoundry/chroma`). Choose another folder:

```python
"vector_store": {"provider": "chroma", "collection_name": "my_docs",
                 "persist_directory": "./chroma_data"}
```

## Chroma server

Start a server:

```bash
docker run -d -p 8000:8000 -v ./chroma-data:/data chromadb/chroma
```

```python
"vector_store": {"provider": "chroma", "collection_name": "my_docs",
                 "host": "localhost", "port": 8000}
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `collection_name` | `ragfoundry` | Chroma collection |
| `persist_directory` | `persist_dir/chroma` | Folder for local mode |
| `host` | | Server host; switches to client/server mode |
| `port` | `8000` | Server port |

## Complete example

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "handbook"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
})
with RAGPipeline(config) as rag:
    rag.ingest("docs")
    print(rag.query("How many holiday days do we get?").answer)
```

## Notes

- Local mode is meant for one process at a time. If several processes or servers need the same
  index, run a Chroma server.
