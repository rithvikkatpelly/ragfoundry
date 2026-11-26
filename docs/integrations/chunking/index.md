# Chunking strategies

Chunking splits each document into passages small enough to search precisely and to fit in the
LLM's prompt. Choose a strategy with `chunking.strategy`:

```python
"chunking": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200}
```

| Strategy | How it splits | Ingest cost | Use when |
|---|---|---|---|
| [`recursive`](recursive.md) | At paragraphs, then lines, then words | None | Default; almost all documents |
| [`fixed`](fixed.md) | Exact windows of `chunk_size` | None | Text with no structure; reproducible sizes |
| [`semantic`](semantic.md) | Where the topic changes, measured with embeddings | Embedding calls | Long documents covering many topics |
| [`llm`](llm.md) | Where an LLM says the topic changes | LLM calls | Highest-quality boundaries, small corpora |

Settings shared by all strategies:

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | `1000` | Maximum chunk length |
| `chunk_overlap` | `200` | Length repeated between neighbouring chunks; must be less than `chunk_size` |
| `length_unit` | `characters` | `characters` or `tokens` (needs the `tokens` extra) |
| `encoding_name` | `cl100k_base` | tiktoken encoding used when `length_unit` is `tokens` |

`chunk_size` is an upper bound for every strategy. Each chunk keeps its document's metadata and
gains `chunk_id` and `start_index`.

## Try a strategy without indexing

```python
from langchain_core.documents import Document
from ragfoundry import chunk_documents
from ragfoundry.config import ChunkingConfig

text = open("handbook.md").read()
chunks = chunk_documents(
    [Document(page_content=text, metadata={"source": "handbook.md"})],
    ChunkingConfig(strategy="recursive", chunk_size=500, chunk_overlap=50),
)
for chunk in chunks[:3]:
    print(len(chunk.page_content), repr(chunk.page_content[:80]))
```

Not sure what size to use? See [Choose chunk size](../../how-to/chunk-size.md). Need something
else? [Register your own strategy](../../how-to/custom-components.md#add-a-chunking-strategy).
