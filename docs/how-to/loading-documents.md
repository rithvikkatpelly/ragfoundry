# Load documents

`rag.ingest()` reads, splits and indexes documents in one call. It accepts a file, a folder, a URL,
or a list mixing any of them.

```python
rag.ingest("report.pdf")                        # one file
rag.ingest("docs")                              # every supported file in a folder, recursively
rag.ingest("https://example.com/handbook")      # a web page
rag.ingest(["docs", "faq.md", "https://example.com/pricing"])
```

Supported files: `.pdf`, `.docx`, `.txt`, `.md`, `.markdown`, `.csv`, `.xlsx`, `.xlsm`, `.xls`.
Each has its own page under [Document loaders](../integrations/loaders/index.md).

## The ingest report

```python
report = rag.ingest("docs")
print(report.to_dict())
```

```python
{
    "documents_loaded": 12,   # pages, rows or files read
    "chunks_total": 85,       # chunks those produced
    "chunks_added": 85,       # newly embedded and stored
    "chunks_unchanged": 0,    # already indexed, skipped
    "chunks_removed": 0,      # stale chunks deleted
    "failures": [],           # files that couldn't be read (with on_error: skip)
}
```

## Choose which files in a folder

`loading.glob` selects files inside folders:

```python
"loading": {"glob": "**/*.pdf"}        # only PDFs, in all subfolders
"loading": {"glob": "*.md"}            # only Markdown, top level only
```

Unsupported file types in a folder are ignored. A folder with no supported files is an error.

## Handle unreadable files

By default the first bad file (corrupt, encrypted, scanned PDF without text) stops the ingest with
an `IngestionError`. To skip bad files and carry on:

```python
"loading": {"on_error": "skip"}
```

```python
report = rag.ingest("docs")
for failure in report.failures:
    print(f"Skipped {failure.source}: {failure.error}")
```

A file that fails is left as it was in the index: its existing chunks are not deleted.

## Text encoding

Text, Markdown and CSV files are read as UTF-8. For other encodings:

```python
"loading": {"encoding": "latin-1"}
```

## Metadata

Every chunk carries metadata you can use for citations and [filters](metadata-filters.md):

| Key | Value |
|---|---|
| `source` | Absolute file path, or the URL |
| `file_type` | `pdf`, `docx`, `md`, `csv`, `web` ... |
| `chunk_id` | Stable id of the chunk |
| `start_index` | Character offset of the chunk in its document |
| `page` | PDF page, starting at 0 |
| `row` | CSV or Excel row |
| `sheet` | Excel sheet name |
| `title` | Web page title |

## Look at documents before indexing

```python
from ragfoundry import load_sources
from ragfoundry.config import LoadingConfig

result = load_sources(["docs"], LoadingConfig())
for doc in result.documents[:3]:
    print(doc.metadata, doc.page_content[:100])
```

## Other file types

Register a loader for any extension. See [Custom components](custom-components.md#add-a-file-type).
