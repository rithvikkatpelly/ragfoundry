# Evaluate quality

`rag.evaluate()` scores your pipeline against a set of test questions. Use it to compare settings
and to catch regressions. For a guided walkthrough, see
[Evaluate and improve your RAG](../tutorials/evaluate-and-improve.md).

## Test set format

JSON Lines, one question per line (a `.json` file holding a list also works):

```json
{"question": "How long do refunds take?", "reference_answer": "Five business days.", "relevant_sources": ["refunds.md"]}
{"question": "Who approves travel?", "reference_contexts": ["Travel must be approved by a line manager"]}
```

| Field | Meaning |
|---|---|
| `question` | Required |
| `relevant_sources` | Files (name, path ending, or full path) or URLs that should be retrieved |
| `reference_contexts` | Passages that a correct chunk should contain, or be contained in |
| `relevant_ids` | Exact `chunk_id` values |
| `reference_answer` | A correct answer, for `answer_correctness` and `context_recall` |

Retrieval metrics need at least one of `relevant_sources`, `reference_contexts` or `relevant_ids`.

Or build samples in code:

```python
from ragfoundry import EvalSample

samples = [EvalSample(question="How long do refunds take?",
                      reference_answer="Five business days.",
                      relevant_sources=["refunds.md"])]
report = rag.evaluate(samples)
```

## Run

```python
report = rag.evaluate("eval.jsonl", mode="retrieval")    # fast; no LLM calls
report = rag.evaluate("eval.jsonl", mode="generation")   # answer quality, judged by an LLM
report = rag.evaluate("eval.jsonl", mode="all")          # both (default)

for metric, stats in sorted(report.summary.items()):
    print(f"{metric:<20} {stats['mean']:.3f}  (n={stats['count']})")

report.save("report.json")       # every question with its answer, passages, scores and errors
```

`count` is how many questions a metric was scored on; a metric is skipped where it doesn't apply.

## Retrieval metrics

Computed at each `k` in `evaluation.k_values` (default `[1, 3, 5]`):

| Metric | Meaning |
|---|---|
| `hit_rate@k` | 1 if any relevant document is in the top *k* |
| `precision@k` | Share of the top *k* that is relevant |
| `recall@k` | Share of the relevant documents found in the top *k* |
| `ndcg@k` | Rewards relevant results ranked higher |
| `mrr` | 1 / rank of the first relevant result |
| `map` | Mean average precision over all relevant results |

Several chunks from one relevant document each count towards precision, but the document counts
once towards recall, MAP and nDCG.

## Generation metrics

Judged by an LLM:

| Metric | Measures | Needs |
|---|---|---|
| `faithfulness` | Share of the answer's claims supported by the passages (low = hallucination) | passages |
| `answer_relevance` | How directly the answer addresses the question | |
| `context_relevance` | Share of retrieved passages useful for the question | passages |
| `answer_correctness` | Agreement with the reference answer | `reference_answer` |
| `context_recall` | Share of the reference answer's claims found in the passages | `reference_answer` |

## Settings

```python
"evaluation": {
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},   # judge; defaults to the pipeline LLM
    "k_values": [1, 3, 5, 10],
    "max_concurrency": 4,                                      # questions evaluated in parallel
}
```

A judge model different from the answering model gives fairer scores. Raise `max_concurrency`
to finish faster, within your provider's rate limits.

## RAGAS

To score generation with [RAGAS](https://docs.ragas.io) metrics instead:

```bash
pip install "ragfoundry[ragas]"
```

```python
report = rag.evaluate("eval.jsonl", mode="generation", backend="ragas")
```

## Metric functions on their own

```python
from ragfoundry.evaluation import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k

recall_at_k(["a", "x", "b"], {"a", "b"}, k=3)     # 1.0
```
