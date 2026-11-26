# Evaluate and improve your RAG

"It seems to work" isn't enough to know whether a change made your RAG better. In this tutorial
you will write a small test set, measure a baseline, then try improvements and keep only the ones
that help.

**Time:** 30 minutes · **Needs:** a [Groq API key](../get-started/api-keys.md) and some documents
in `docs/`.

## 1. Write a test set

Create `eval.jsonl` with one question per line. For each, say which file should be found and,
optionally, what a correct answer is:

```json title="eval.jsonl"
{"question": "How long do refunds take?", "relevant_sources": ["company.md"], "reference_answer": "Five business days."}
{"question": "Who approves refunds over 500 dollars?", "relevant_sources": ["company.md"], "reference_answer": "A finance manager."}
{"question": "How many holiday days do employees get?", "relevant_sources": ["company.md"], "reference_answer": "25 days per year."}
{"question": "When is the office open?", "relevant_sources": ["company.md"], "reference_answer": "8am to 6pm, Monday to Friday."}
```

Tips:

- Use real questions your users ask, worded the way they word them.
- 20 to 50 questions is enough to see real differences; 5 is enough to start.
- `relevant_sources` can be a file name, the end of a path, or a URL.
- Include a few questions whose answers are *not* in your documents, without
  `relevant_sources`. Retrieval metrics skip them; check in `report.json` that their answers
  decline politely.

## 2. Measure the baseline

```python title="evaluate.py"
from ragfoundry import RAGConfig, RAGPipeline

BASE = {
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "eval"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0},
    "evaluation": {"k_values": [1, 3, 5]},
}


def score(name: str, **changes) -> None:
    with RAGPipeline(RAGConfig.model_validate({**BASE, **changes})) as rag:
        rag.ingest("docs")
        report = rag.evaluate("eval.jsonl", mode="retrieval")
        s = report.summary
        print(f"{name:<12} hit@1={s['hit_rate@1']['mean']:.2f}  "
              f"recall@5={s['recall@5']['mean']:.2f}  mrr={s['mrr']['mean']:.2f}")


score("baseline")
```

`mode="retrieval"` measures only whether the right passages are found. It needs no LLM, so it's
fast and free. Retrieval is where most RAG problems are.

What the numbers mean:

| Metric | Question it answers |
|---|---|
| `hit_rate@k` | For how many questions was at least one right document in the top *k*? |
| `recall@k` | What share of the right documents were in the top *k*? |
| `mrr` | How high was the first right document, on average? 1.0 means always first. |

## 3. Try improvements one at a time

```python
score("hybrid", retrieval={"hybrid": {"enabled": True}})
score("rerank", retrieval={
    "hybrid": {"enabled": True},
    "rerank": {"provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
})
score("small-chunks", chunking={"chunk_size": 400, "chunk_overlap": 80},
      vector_store={"provider": "chroma", "collection_name": "eval_small"})
```

!!! note
    Changing chunking needs a new `collection_name` (or deleting `.ragfoundry/`), because the old
    chunks would otherwise stay in the index alongside the new ones.

Keep a change only if it raises the scores you care about. More features aren't always better:
each adds latency, and sometimes cost.

## 4. Check the answers

Once retrieval is good, measure the answers too:

```python
with RAGPipeline(RAGConfig.model_validate({**BASE, "retrieval": {"hybrid": {"enabled": True}}})) as rag:
    rag.ingest("docs")
    report = rag.evaluate("eval.jsonl", mode="all")
    for metric in ("faithfulness", "answer_relevance", "answer_correctness"):
        print(metric, round(report.summary[metric]["mean"], 2))
    report.save("report.json")
```

| Metric | Low score means |
|---|---|
| `faithfulness` | The answer says things the passages don't support (hallucination). Try a stronger LLM or a stricter [system prompt](../how-to/prompts.md). |
| `answer_relevance` | The answer doesn't address the question. Check the prompt and the model. |
| `answer_correctness` | The answer disagrees with your reference. Check retrieval first. |

These are judged by an LLM, so they cost API calls and vary a little between runs. For fairer
scores, judge with a different model from the one answering:

```python
"evaluation": {"llm": {"provider": "openai", "model": "gpt-4o-mini"}}
```

## 5. Read the failures

Averages hide the details. `report.json` lists every question with its scores; look at the worst
ones and print what was retrieved:

```python
for doc in rag.retrieve("Who approves refunds over 500 dollars?"):
    print(doc.metadata["source"], "|", doc.page_content[:100])
```

Common causes and fixes are in [Troubleshooting](../reference/troubleshooting.md).

## Next

[Evaluation reference](../how-to/evaluation.md) · [Hybrid search](../how-to/hybrid-search.md) ·
[Reranking](../how-to/reranking.md)
