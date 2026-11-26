# Metadata filters

`where` limits a search to chunks whose metadata matches exactly. Use it to search one document, one
file type, or your own categories.

```python
rag.query("What is the notice period?", where={"file_type": "pdf"})
rag.retrieve("brake parts", where={"file_type": "csv"})
chat.ask("Summarise it", where={"source": "/home/me/docs/contract.pdf"})
```

Several keys must all match:

```python
rag.query("What changed?", where={"source": "/home/me/docs/policy.pdf", "page": 3})
```

## Built-in metadata

| Key | Example values |
|---|---|
| `source` | `/home/me/docs/policy.pdf`, `https://example.com/faq` |
| `file_type` | `pdf`, `docx`, `txt`, `md`, `csv`, `xlsx`, `web` |
| `page` | PDF page number, starting at 0 |
| `sheet` | Excel sheet name |

!!! warning "Sources are absolute paths"
    Files are stored with their full path, even if you ingested `docs/policy.pdf`. Build the value
    with `str(Path("docs/policy.pdf").resolve())`, or copy it from `rag.sources()`.

## Your own metadata

Add fields such as department, customer or year with a [custom loader](custom-components.md#add-a-file-type)
that sets them in each document's metadata, then filter on them:

```python
rag.query("What is our travel budget?", where={"department": "finance"})
```

## Notes

- Matching is exact equality. Ranges ("year after 2020") and "any of" lists aren't supported
  through `where`.
- Filters are applied by the vector store itself, and to hybrid keyword search as well.
- ragfoundry translates `where` into each vector store's own filter syntax, so the same filter
  works whichever store you use.
