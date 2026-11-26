# Choose chunk size

Chunk size is the setting that most affects retrieval quality, after the embedding model.

```python
"chunking": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200}
```

## The trade-off

| Smaller chunks (200 to 500 characters) | Larger chunks (1,000 to 2,000 characters) |
|---|---|
| Each chunk is about one thing, so search is precise | Each chunk carries more context around a fact |
| A fact may be split from the sentence that explains it | Several topics in one chunk blur its meaning |
| More chunks: slower ingest, more storage | Fewer chunks, but each uses more of the context window |
| Good for FAQs, short answers, reference tables | Good for narrative text, legal and technical explanations |

## Starting points

| Content | `chunk_size` | `chunk_overlap` |
|---|---|---|
| General documents, manuals, handbooks | 1000 | 200 |
| FAQs, support articles, short entries | 400 | 50 |
| Legal contracts, research papers | 1500 | 300 |
| Code or structured text | 800 | 100 |

CSV and Excel rows are already one document each, and rows are usually far shorter than
`chunk_size`, so they stay whole.

## Overlap

`chunk_overlap` repeats the end of one chunk at the start of the next, so a sentence cut at a
boundary appears complete in at least one chunk. 10 to 20% of `chunk_size` is typical. It must be
smaller than `chunk_size`.

## Characters or tokens

Sizes are in characters by default. To size by LLM tokens (about 4 characters each in English):

```python
"chunking": {"chunk_size": 256, "chunk_overlap": 32, "length_unit": "tokens"}
```

This needs the `tokens` extra (`pip install "ragfoundry[tokens]"`).

## Measure, don't guess

Try two or three sizes against a test set and keep the best. Use a new `collection_name` for each,
because a re-chunked document produces different chunks:

```python
for size in (400, 800, 1200):
    config = RAGConfig.model_validate({
        **base,
        "chunking": {"chunk_size": size, "chunk_overlap": size // 5},
        "vector_store": {"provider": "chroma", "collection_name": f"size_{size}"},
    })
    with RAGPipeline(config) as rag:
        rag.ingest("docs")
        recall = rag.evaluate("eval.jsonl", mode="retrieval").summary["recall@5"]["mean"]
        print(size, round(recall, 3))
```

See [Evaluate and improve your RAG](../tutorials/evaluate-and-improve.md).

## Changing chunking on an existing index

If you change `chunk_size`, `chunk_overlap` or `strategy` and re-ingest into the same collection,
each re-read file's old chunks are replaced by the new ones. To start completely clean, use a new
`collection_name` or delete `persist_dir` and the vector store's data.

## Other strategies

`recursive` suits almost everything. For alternatives (exact windows, topic-aware semantic
splitting, LLM-chosen boundaries) see [Chunking strategies](../integrations/chunking/index.md).
