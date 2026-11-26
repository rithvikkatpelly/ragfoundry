# Qdrant

[Qdrant](https://qdrant.tech) is an open-source vector database. Use local mode while developing,
then point the same configuration at a Qdrant server or Qdrant Cloud.

| | |
|---|---|
| Extra | `qdrant` |
| Setup | None locally; a server or cloud cluster in production |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[qdrant]"
```

## Local mode (no server)

```python
"vector_store": {"provider": "qdrant", "collection_name": "my_docs"}
```

Data is saved in `persist_dir/qdrant`, or in `path` if you set it.

## Server

```bash
docker run -d -p 6333:6333 -v ./qdrant-data:/qdrant/storage qdrant/qdrant
```

```python
"vector_store": {"provider": "qdrant", "url": "http://localhost:6333", "collection_name": "my_docs"}
```

## Qdrant Cloud

```python
"vector_store": {
    "provider": "qdrant",
    "url": "https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333",
    "api_key": "...",                     # in YAML: ${QDRANT_API_KEY}
    "collection_name": "my_docs",
}
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `collection_name` | `ragfoundry` | Qdrant collection, created with the right size and cosine distance |
| `url` | | Server or cloud URL; switches from local mode |
| `api_key` | | Required for Qdrant Cloud |
| `path` | `persist_dir/qdrant` | Folder for local mode |

## Notes

- Local mode allows only one process at a time to open the folder. Use a server when several
  processes share the index.
