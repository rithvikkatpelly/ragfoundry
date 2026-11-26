# Semantic chunking

Splits where the *meaning* changes. Each sentence is embedded together with its neighbours, the
similarity of consecutive sentences is measured, and chunks are cut at the biggest drops, the
points where the text moves to a new topic.

```python
"chunking": {
    "strategy": "semantic",
    "chunk_size": 1500,
    "chunk_overlap": 100,
    "semantic": {"breakpoint_percentile": 95, "buffer_size": 1},
}
```

It uses the pipeline's own [embedding model](../embeddings/index.md), so there's nothing extra to
install, but every sentence is embedded during ingestion.

## Options

| Setting | Default | Meaning |
|---|---|---|
| `semantic.breakpoint_percentile` | `95` | Cut where the similarity drop is in the top (100 − this)%. Lower = more, smaller chunks |
| `semantic.buffer_size` | `1` | Neighbouring sentences embedded with each sentence, to smooth noise |
| `chunk_size` | `1000` | Upper limit; longer sections are split recursively |
| `chunk_overlap` | `200` | Overlap used when a long section is split |

## When to use it

- Long documents that move between topics without clear headings: reports, transcripts, books.
- When recursive chunking keeps mixing two topics in one chunk.

## Cost

Each sentence is embedded once during ingestion, in addition to the chunks themselves. With local
Hugging Face embeddings that only costs time; with hosted embeddings it costs roughly double the
usual embedding spend.

Documents with fewer than three sentences are kept whole (or split by size).
