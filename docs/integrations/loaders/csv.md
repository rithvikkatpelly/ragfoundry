# CSV

Reads each row of a CSV file as its own document, written as `column: value` lines so the meaning
of each value stays attached to it.

| | |
|---|---|
| Extensions | `.csv` |
| Extra | none |
| One document per | row |
| Metadata | `source`, `file_type: csv`, `row` (starting at 0) |

A file like this:

```text title="parts.csv"
part_number,description,price_usd
ZT-9902,brake caliper,89.00
```

becomes a document:

```text
part_number: ZT-9902
description: brake caliper
price_usd: 89.00
```

## Use

```python
rag.ingest("parts.csv")
result = rag.query("How much does the brake caliper cost?")
```

## Tips

- Turn on [hybrid search](../../how-to/hybrid-search.md) for tables with codes, ids or names:
  keyword search finds `ZT-9902` exactly.
- The first row must be the header.
- RAG is good at "look up this row" questions. It can't compute totals, averages or counts over
  a whole table, because it only sees a few rows per question. For analytics, query the data
  with SQL or pandas instead.
- Files are read as UTF-8; set `loading.encoding` for others.
