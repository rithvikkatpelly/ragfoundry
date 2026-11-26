# Pinecone

[Pinecone](https://www.pinecone.io) is a fully managed, serverless vector database. There are no
servers to run.

| | |
|---|---|
| Extra | `pinecone` |
| Key | `PINECONE_API_KEY` ([get one](https://app.pinecone.io)) |
| Setup | Create an index in the Pinecone console first |
| Filters | Yes |

## 1. Create an index

In the [Pinecone console](https://app.pinecone.io), create an index with:

- **Dimensions** equal to your embedding model's: 384 for `all-MiniLM-L6-v2` and
  `bge-small-en-v1.5`, 768 for `bge-base-en-v1.5`, 1536 for `text-embedding-3-small`, 3072 for
  `text-embedding-3-large`.
- **Metric**: cosine.

## 2. Install

```bash
pip install "ragfoundry[pinecone]"
export PINECONE_API_KEY="..."
```

## 3. Configure

=== "Python"

    ```python
    "vector_store": {
        "provider": "pinecone",
        "index_name": "my-index",
        "collection_name": "handbook",      # used as the Pinecone namespace
    }
    ```

=== "YAML"

    ```yaml
    vector_store:
      provider: pinecone
      index_name: my-index
      collection_name: handbook
      api_key: ${PINECONE_API_KEY}
    ```

## Options

| Setting | Default | Meaning |
|---|---|---|
| `index_name` | required | The index you created |
| `collection_name` | `ragfoundry` | Namespace inside the index, so several datasets can share one index |
| `api_key` | `PINECONE_API_KEY` | API key |

## Notes

- A dimension mismatch between index and embedding model fails on the first ingest. Create a new
  index with the right dimension.
- Pinecone is eventually consistent: chunks may take a few seconds to become searchable after
  ingest.
