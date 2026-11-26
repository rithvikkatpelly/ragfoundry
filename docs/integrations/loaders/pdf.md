# PDF

Reads the text of each page of a PDF using [pypdf](https://pypdf.readthedocs.io).

| | |
|---|---|
| Extensions | `.pdf` |
| Extra | `pdf` |
| One document per | page |
| Metadata | `source`, `file_type: pdf`, `page` (starting at 0), plus PDF info such as `title`, `author`, `total_pages` when present |

## Install

```bash
pip install "ragfoundry[pdf]"
```

## Use

```python
rag.ingest("report.pdf")
rag.ingest("contracts/")       # every PDF (and other supported file) in the folder
```

## Show page numbers in citations

```python
from pathlib import Path

result = rag.query("What is the termination clause?")
for number, doc in enumerate(result.documents, start=1):
    print(f"[{number}] {Path(doc.metadata['source']).name}, page {doc.metadata['page'] + 1}")
```

## Search a single page range or file

```python
from pathlib import Path

contract = str(Path("contracts/acme.pdf").resolve())
rag.query("What is the notice period?", where={"source": contract})
rag.query("Summarise this page", where={"source": contract, "page": 4})
```

## Limitations

- **Scanned PDFs** are images with no text layer. They fail with
  `No extractable text in ...`. Run OCR first, for example with
  [OCRmyPDF](https://ocrmypdf.readthedocs.io): `ocrmypdf scanned.pdf searchable.pdf`.
- Blank pages are skipped.
- Tables are read as plain text in reading order; complex layouts may come out jumbled.
- Password-protected PDFs can't be read.
