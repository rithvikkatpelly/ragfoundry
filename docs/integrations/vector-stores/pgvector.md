# PGVector

[pgvector](https://github.com/pgvector/pgvector) adds vector search to PostgreSQL. If your
application already uses Postgres, your vectors can live next to the rest of your data.

| | |
|---|---|
| Extra | `pgvector` |
| Setup | PostgreSQL with the `vector` extension |
| Filters | Yes |

## Install

```bash
pip install "ragfoundry[pgvector]"
```

Start Postgres with pgvector if you don't have one:

```bash
docker run -d -p 5432:5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag pgvector/pgvector:pg16
```

Managed services such as Supabase, Neon, AWS RDS and Google Cloud SQL support pgvector too; enable
the `vector` extension in their console.

## Configure

=== "Python"

    ```python
    "vector_store": {
        "provider": "pgvector",
        "connection": "postgresql+psycopg://rag:rag@localhost:5432/rag",
        "collection_name": "handbook",
    }
    ```

=== "YAML"

    ```yaml
    vector_store:
      provider: pgvector
      connection: ${DATABASE_URL}
      collection_name: handbook
    ```

The connection string uses the `postgresql+psycopg://` scheme (SQLAlchemy with psycopg 3).

## Options

| Setting | Default | Meaning |
|---|---|---|
| `connection` | required | Connection string |
| `collection_name` | `ragfoundry` | Collection; several can share one database |

## Notes

- Tables are created automatically on first ingest. The database user needs permission to create
  tables, and the `vector` extension must be available.
- Metadata is stored as JSONB, so [metadata filters](../../how-to/metadata-filters.md) work.
