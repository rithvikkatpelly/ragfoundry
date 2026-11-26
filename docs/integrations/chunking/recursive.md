# Recursive chunking

The default strategy. It tries to split at the most natural boundary that keeps chunks under
`chunk_size`: first blank lines (paragraphs), then line breaks, then spaces, and only as a last
resort in the middle of a word. Paragraphs and sentences usually stay whole.

```python
"chunking": {
    "strategy": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 200,
}
```

Built on LangChain's `RecursiveCharacterTextSplitter`.

## Options

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | `1000` | Maximum chunk length |
| `chunk_overlap` | `200` | Overlap between neighbouring chunks |
| `length_unit` | `characters` | Or `tokens` |
| `separators` | `["\n\n", "\n", " ", ""]` | Boundaries to try, in order |

## Custom separators

Split Markdown at headings first:

```python
"chunking": {
    "strategy": "recursive",
    "chunk_size": 1500,
    "chunk_overlap": 100,
    "separators": ["\n## ", "\n### ", "\n\n", "\n", " ", ""],
}
```

## Sizing by tokens

```python
"chunking": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 50,
             "length_unit": "tokens"}
```

Requires `pip install "ragfoundry[tokens]"`.
