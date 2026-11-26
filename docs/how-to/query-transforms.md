# Query transforms

Users write short, vague or oddly worded questions. Query transforms ask the LLM to improve the
search query before retrieval. Each strategy adds one LLM call per question.

```python
"retrieval": {
    "query_transform": {
        "strategies": ["multi_query"],   # any of: rewrite, multi_query, hyde
        "num_queries": 3,                # used by multi_query
    },
}
```

## Strategies

| Strategy | What it does | Helps when |
|---|---|---|
| `rewrite` | Replaces the question with a clearer, self-contained search query | Questions are messy, chatty or full of typos |
| `multi_query` | Adds `num_queries` differently worded queries; all are searched and the results merged | Your documents use different words than your users |
| `hyde` | The LLM writes a short *hypothetical* answer, which is used for vector search | Questions are short but answers are long passages |

You can combine them, for example `["rewrite", "multi_query"]`. The queries actually used are in
`result.queries`:

```python
result = rag.query("price of ZT-9902?")
print(result.queries)
# ['price of ZT-9902?', 'What is the cost of part ZT-9902?', 'How much does ZT-9902 cost?', ...]
```

## Notes

- The LLM answering the question always sees the original question, not the transformed one.
- HyDE passages are used for vector search only, never keyword search, because a made-up passage
  can contain wrong keywords.
- In a [chat](chat-history.md), transforms start from the rewritten standalone question.
- Transforms add latency. Measure with an [evaluation](evaluation.md) whether they help your data
  before leaving them on.
