# Document loaders

A loader turns a file or web page into documents: text plus metadata. ragfoundry chooses the loader
from the file extension (or `http(s)://` for URLs), so you just pass paths to `ingest`:

```python
rag.ingest(["contracts/", "faq.md", "prices.xlsx", "https://example.com/help"])
```

| Loader | Extensions | Extra | One document per | Metadata |
|---|---|---|---|---|
| [PDF](pdf.md) | `.pdf` | `pdf` | page | `page` |
| [Word](docx.md) | `.docx` | `docx` | file | |
| [Text and Markdown](text.md) | `.txt`, `.md`, `.markdown` | none | file | |
| [CSV](csv.md) | `.csv` | none | row | `row` |
| [Excel](excel.md) | `.xlsx`, `.xlsm`, `.xls` | `excel` | row | `sheet`, `row` |
| [Web pages](web.md) | URLs | `web` | page | `title` |

Every document also gets `source` (absolute path or URL) and `file_type`. Install all loaders at
once with the `loaders` extra:

```bash
pip install "ragfoundry[loaders]"
```

Settings shared by all loaders, under `loading`:

| Setting | Default | Meaning |
|---|---|---|
| `on_error` | `raise` | `skip` to report unreadable files and continue |
| `encoding` | `utf-8` | Encoding for text, Markdown and CSV files |
| `glob` | `**/*` | Which files to take from folders |
| `web` | | Crawling options; see [Web pages](web.md) |

Need another format? [Register a loader](../../how-to/custom-components.md#add-a-file-type).
