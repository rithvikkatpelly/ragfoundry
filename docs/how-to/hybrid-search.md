# Hybrid search

Vector search finds passages by *meaning*. It's good at "how do I get my money back?" matching a
passage about refunds, but weaker at exact strings: product codes, names, error numbers, rare
terms. Keyword search (BM25) is the opposite. Hybrid search runs both and merges the results.

```python
"retrieval": {"hybrid": {"enabled": True}}
```

That's usually all you need. It's cheap (the keyword index is in memory) and rarely hurts.

## How results are merged

Both searches return a ranked list. They're combined with **reciprocal rank fusion (RRF)**: each
passage scores `weight / (rrf_k + rank)` in each list it appears in, and the scores are added.
Passages ranked well by both methods rise to the top.

```python
"retrieval": {
    "fetch_k": 20,                 # candidates taken from each search before merging
    "hybrid": {
        "enabled": True,
        "rrf_k": 60,               # higher = flatter weighting of ranks; 60 is standard
        "vector_weight": 1.0,
        "bm25_weight": 1.0,        # raise above 1 to favour exact keyword matches
    },
}
```

## When it helps most

- Part numbers, SKUs, ticket ids: `"What does part ZT-9902 cost?"`
- Names of people, products, laws, functions
- Acronyms and jargon the embedding model doesn't know
- Short queries of one or two words

## How it works with other features

- With [query transforms](query-transforms.md), every query variant is searched both ways and all
  lists are fused together. HyDE passages go to vector search only.
- With [reranking](reranking.md), fusion produces the candidates and the reranker orders them.
- [Metadata filters](metadata-filters.md) apply to both searches.

## Limits

The keyword index is built from the chunk record in `persist_dir` and scores every chunk per
query. That's fast up to a few hundred thousand chunks. Every process that queries must use the
same `persist_dir` as the one that ingested.
