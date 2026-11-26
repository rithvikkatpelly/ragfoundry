# Text and Markdown

Reads plain text and Markdown files as they are.

| | |
|---|---|
| Extensions | `.txt`, `.md`, `.markdown` |
| Extra | none |
| One document per | file |
| Metadata | `source`, `file_type` (`txt`, `md` or `markdown`) |

## Use

```python
rag.ingest("notes.txt")
rag.ingest("wiki/")                         # a folder of Markdown files
```

## Encoding

Files are read as UTF-8. For other encodings:

```python
"loading": {"encoding": "latin-1"}
```

## Tips

- Markdown headings, lists and code are kept as text. The [recursive](../chunking/recursive.md)
  chunker splits on blank lines first, so sections tend to stay together.
- To split exactly at headings, register a
  [custom chunker](../../how-to/custom-components.md#add-a-chunking-strategy).
- Only these extensions are picked up from folders. For other text formats (`.rst`, `.log`),
  [register a loader](../../how-to/custom-components.md#add-a-file-type):

    ```python
    from langchain_community.document_loaders import TextLoader
    from ragfoundry import register_loader

    @register_loader(".rst", ".log")
    def plain_text(path, loading_config):
        return TextLoader(str(path), encoding=loading_config.encoding)
    ```
