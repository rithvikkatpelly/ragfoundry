# Fixed chunking

Cuts text into windows of exactly `chunk_size`, each starting `chunk_size - chunk_overlap` after the
previous one. It ignores sentences and paragraphs, so chunks often start or end mid-word.

```python
"chunking": {"strategy": "fixed", "chunk_size": 500, "chunk_overlap": 100}
```

With `chunk_size: 500` and `chunk_overlap: 100`, chunks cover characters 0 to 500, 400 to 900,
800 to 1,300, and so on.

## Options

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | `1000` | Window length |
| `chunk_overlap` | `200` | How much consecutive windows share |
| `length_unit` | `characters` | With `tokens`, windows are exact token counts (tiktoken) |

## When to use it

- Text without useful structure: transcripts, OCR output, logs.
- Experiments where every chunk must be the same size.

For ordinary documents, [recursive](recursive.md) chunking gives cleaner passages and usually
better retrieval.
