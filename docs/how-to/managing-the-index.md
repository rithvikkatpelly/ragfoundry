# Keep the index up to date

## Re-ingest when documents change

Run the same `ingest` call again:

```python
report = rag.ingest("docs")
print(report.chunks_added, report.chunks_unchanged, report.chunks_removed)
# 3 82 1
```

ragfoundry keeps a record of every chunk (in `persist_dir`). On each run it:

- **skips** chunks that are already indexed and unchanged,
- **embeds and adds** new or edited chunks,
- **removes** chunks that no longer exist in a file it re-read.

So re-running over a large folder is cheap: only the changes cost embedding calls. If a run is
interrupted, run it again and it picks up where it stopped.

Each chunk's id is derived from its file and its text, so editing one paragraph usually re-embeds
only the chunks around that edit.

## See what's indexed

```python
for source, chunks in rag.sources().items():
    print(chunks, source)
```

## Remove a document

```python
rag.delete_source("docs/old-policy.pdf")                 # relative or absolute path
rag.delete_source("https://example.com/retired-page")    # URLs must match exactly
```

It returns the number of chunks removed. Deleting a file from disk doesn't remove it from the
index by itself, because ingest only looks at the files you give it; call `delete_source` too.

## Sync a folder, including deletions

```python
from pathlib import Path

folder = Path("docs").resolve()
rag.ingest(folder)
on_disk = {str(p) for p in folder.rglob("*") if p.is_file()}
for source in rag.sources():
    if source.startswith(str(folder)) and source not in on_disk:
        rag.delete_source(source)
```

## Schedule it

Run a sync script from cron, a scheduled CI job, or your task queue, using the same configuration
(vector store, collection and `persist_dir`) as your application. If the application keeps
answering while a sync runs in another process, use a server vector store such as
[Qdrant](../integrations/vector-stores/qdrant.md) or [PGVector](../integrations/vector-stores/pgvector.md);
local file-based stores are designed for one process at a time.

## Change the embedding model

A collection remembers which embedding model filled it. Switching model on an existing collection
raises an error, because vectors from two models can't be compared. To switch:

1. Set a new `vector_store.collection_name`.
2. Ingest everything again.
3. Point your application at the new collection, then delete the old one.

## Start over

Delete `persist_dir` (default `.ragfoundry/`) and, for server vector stores, the collection in
the store. The next ingest rebuilds everything.
