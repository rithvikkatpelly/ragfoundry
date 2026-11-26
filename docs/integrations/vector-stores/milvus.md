# Milvus

[Milvus](https://milvus.io) is an open-source vector database designed for very large collections.
**Milvus Lite** runs from a local file for development; the same code then works with a Milvus
server or Zilliz Cloud.

| | |
|---|---|
| Extra | `milvus` (includes Milvus Lite) |
| Setup | None for Milvus Lite |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[milvus]"
```

## Milvus Lite (local file)

```python
"vector_store": {"provider": "milvus", "collection_name": "my_docs"}
```

Data is saved to `persist_dir/milvus.db`. Milvus Lite runs on macOS and Linux.

## Milvus server or Zilliz Cloud

Install a server with the [Milvus standalone guide](https://milvus.io/docs/install_standalone-docker.md),
then:

```python
"vector_store": {
    "provider": "milvus",
    "uri": "http://localhost:19530",
    "token": None,                       # "user:password", or a Zilliz Cloud API key
    "collection_name": "my_docs",
}
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `uri` | local file | Server or Zilliz Cloud URI; omit for Milvus Lite |
| `token` | | Authentication token |
| `collection_name` | `ragfoundry` | Collection; hyphens become underscores |
