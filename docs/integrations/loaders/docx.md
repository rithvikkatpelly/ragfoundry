# Word (DOCX)

Reads the text of Microsoft Word `.docx` files with [docx2txt](https://github.com/ankushshah89/python-docx2txt).

| | |
|---|---|
| Extensions | `.docx` |
| Extra | `docx` |
| One document per | file |
| Metadata | `source`, `file_type: docx` |

## Install

```bash
pip install "ragfoundry[docx]"
```

## Use

```python
rag.ingest("policies/handbook.docx")
```

The whole file becomes one document, which the [chunker](../chunking/index.md) then splits.

## Limitations

- Old `.doc` files (Word 97-2003) aren't supported. Open them in Word or LibreOffice and save as
  `.docx`, or convert in bulk: `soffice --headless --convert-to docx *.doc`.
- Headers, footers and comments aren't included.
- Images are ignored; text in text boxes and tables is included.
