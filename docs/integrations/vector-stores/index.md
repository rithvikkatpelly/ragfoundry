# Vector stores

The vector store keeps chunk vectors and finds the closest ones to each question. Choose one with
the `vector_store` section:

```python
"vector_store": {"provider": "chroma", "collection_name": "my_docs"}
```

| Store | Runs | Setup | Extra | Good for |
|---|---|---|---|---|
| [Chroma](chroma.md) | Local folder or server | None | `chroma` | Getting started, small to medium apps |
| [FAISS](faiss.md) | Local files | None | `faiss` | Fast single-process search |
| [Qdrant](qdrant.md) | Local folder, server or cloud | None locally | `qdrant` | Dev locally, then the same code in production |
| [Milvus](milvus.md) | Local file, server or cloud | None locally | `milvus` | Very large collections |
| [PGVector](pgvector.md) | PostgreSQL | A Postgres database | `pgvector` | Teams already using Postgres |
| [Weaviate](weaviate.md) | Server or cloud | A Weaviate instance | `weaviate` | Existing Weaviate users |
| [Pinecone](pinecone.md) | Managed cloud | An index in the Pinecone console | `pinecone` | Serverless, no infrastructure |

## Shared settings

| Setting | Default | Meaning |
|---|---|---|
| `provider` | | Which store |
| `collection_name` | `ragfoundry` | Separates datasets in one store (Pinecone: the namespace) |

Each store adds its own connection settings, listed on its page.

## Behaviour common to every store

- Collections are created automatically on first ingest (except Pinecone, where you create the
  index first).
- Similarity is cosine everywhere, so scores and rankings behave the same when you switch store.
- [Metadata filters](../../how-to/metadata-filters.md) are translated into each store's own filter
  syntax.
- Incremental ingestion and [hybrid search](../../how-to/hybrid-search.md) work with every store,
  using the chunk record in `persist_dir`.

## Switching store

Change the `vector_store` section and ingest again. Your documents, chunking and embeddings
settings stay the same.

Another store (Elasticsearch, OpenSearch, MongoDB Atlas, Redis...)? Pass any LangChain
`VectorStore` as `RAGPipeline(config, vector_store=...)`; see
[Custom components](../../how-to/custom-components.md).
