# LLM chunking

Asks the LLM to read the document as numbered sentences and choose where each topic starts.
This gives the most natural chunks, at the cost of LLM calls during ingestion.

```python
"chunking": {
    "strategy": "llm",
    "chunk_size": 1500,
    "chunk_overlap": 100,
    "llm": {"window_chars": 8000},
}
```

It uses the pipeline's `llm`, so an `llm` section is required.

## Options

| Setting | Default | Meaning |
|---|---|---|
| `llm.window_chars` | `8000` | Characters of text sent per LLM call |
| `chunk_size` | `1000` | Upper limit; a topic longer than this is split recursively |
| `chunk_overlap` | `200` | Overlap used when a topic is split |

## How it works

1. The text is split into sentences.
2. Up to `window_chars` of them are sent, numbered, asking which sentences begin a new topic.
3. The last topic of each window is carried into the next, so a topic isn't cut just because a
   window ended.
4. If the LLM's reply can't be understood, that window falls back to
   [recursive](recursive.md) chunking and a warning is logged.

## Cost

About one LLM call per `window_chars` of text: a 100-page document (about 300,000 characters)
takes around 40 calls. Use a fast, cheap model, and use it for small or high-value document sets.
