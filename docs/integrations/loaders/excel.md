# Excel

Reads every sheet of an Excel workbook. Each data row becomes a document written as
`header: value` lines.

| | |
|---|---|
| Extensions | `.xlsx`, `.xlsm` (via openpyxl), `.xls` (via xlrd) |
| Extra | `excel` |
| One document per | row |
| Metadata | `source`, `file_type`, `sheet`, `row` (the spreadsheet row number) |

## Install

```bash
pip install "ragfoundry[excel]"
```

## Use

```python
rag.ingest("inventory.xlsx")

# Only search one sheet
rag.query("Which items are low on stock?", where={"sheet": "Warehouse A"})
```

## How rows are read

- The first non-empty row of each sheet is the header. Empty header cells become `column_1`,
  `column_2` ...
- Empty rows are skipped; empty cells are left out of the row's text.
- Formulas are read as their last saved value, not the formula.
- Whole numbers are written without a trailing `.0`.

## Tips

As with [CSV](csv.md): use hybrid search for codes and names, and use SQL or pandas rather than
RAG for totals and aggregations.
