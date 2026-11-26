# Weaviate

[Weaviate](https://weaviate.io) is an open-source vector database, self-hosted or as Weaviate
Cloud.

| | |
|---|---|
| Extra | `weaviate` |
| Setup | A running Weaviate instance |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[weaviate]"
```

## Self-hosted

```bash
docker run -d -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:1.30.0
```

```python
"vector_store": {
    "provider": "weaviate",
    "url": "http://localhost:8080",
    "grpc_port": 50051,
    "collection_name": "handbook",
}
```

## Weaviate Cloud

Setting `api_key` connects to Weaviate Cloud, with `url` as the cluster URL:

```python
"vector_store": {
    "provider": "weaviate",
    "url": "https://YOUR-CLUSTER.weaviate.cloud",
    "api_key": "...",
    "collection_name": "handbook",
}
```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `url` | `http://localhost:8080` | Weaviate HTTP address, or cluster URL for the cloud |
| `grpc_port` | `50051` | gRPC port for self-hosted Weaviate |
| `api_key` | | Weaviate Cloud API key |
| `collection_name` | `ragfoundry` | Collection name |

## Notes

- Weaviate collection names must start with a capital letter and can't contain hyphens.
  ragfoundry converts the name for you: `my-docs` becomes `My_docs`.
- ragfoundry supplies the vectors, so Weaviate needs no vectorizer module.
