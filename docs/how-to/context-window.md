# Limit the context window

An LLM can only read a certain amount of text per request (its *context window*), and longer
prompts are slower, cost more, and can make answers worse. ragfoundry lets you set a token budget
for the two parts of the prompt that grow: retrieved passages and chat history.

```python
"context_window": {
    "max_context_tokens": 4000,      # retrieved passages; unset = no limit (default)
    "truncate_last_passage": True,   # shorten the passage that crosses the budget (default)
    "max_history_turns": 5,          # chat turns (default 5)
    "max_history_tokens": 2000,      # chat history tokens; unset = no limit (default)
    "token_counter": "approximate",  # approximate (default) | tiktoken
}
```

## Passage budget

Retrieval returns up to `top_k` passages, best first. With `max_context_tokens` set, passages are
added in that order until the budget is reached:

- The most relevant passages always make it in.
- The passage that would cross the limit is **shortened** to fill the space left, ending in `…`,
  and marked `metadata["truncated"] = True`. With `truncate_last_passage: false` it is dropped
  instead.
- Passages after it are left out.

```python
result = rag.query("Summarise the security policy")
print(result.context_tokens)                       # tokens of passages actually sent
for doc in result.documents:                       # exactly what the LLM saw
    print(doc.metadata["source"], doc.metadata.get("truncated", False))
```

## History budget

`max_history_turns` keeps the last N questions with their answers. If `max_history_tokens` is
also set, the oldest of those turns are dropped until the rest fit. See
[Chat with history](chat-history.md).

## Counting tokens

| `token_counter` | How | Needs |
|---|---|---|
| `approximate` | 1 token per 4 characters. Close enough for budgeting English text with most models. | Nothing |
| `tiktoken` | Exact count with an OpenAI encoding (`encoding_name`, default `cl100k_base`) | `pip install "ragfoundry[tokens]"` |

Or pass any function that takes a string and returns its token count, such as your model's own
tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
rag = RAGPipeline(config, token_counter=lambda text: len(tokenizer.encode(text)))
```

## Choosing budgets

The prompt also holds the system prompt (about 100 tokens), the question, and room for the answer
(`llm.max_tokens`). A safe rule is to give passages about half the model's context window and
history about a fifth.

| Model context window | `max_context_tokens` | `max_history_tokens` |
|---|---|---|
| 8,000 tokens (small local models) | 3,000 | 1,000 |
| 32,000 tokens | 12,000 | 4,000 |
| 128,000 tokens and up | 20,000 | 8,000 |

Bigger isn't automatically better: beyond a few thousand tokens of passages, answers often don't
improve, and cost and latency keep rising. Tune with an [evaluation](evaluation.md), watching
`result.context_tokens` and the `truncated` flag.

## Use the functions directly

```python
from ragfoundry import pack_documents, window_history
from ragfoundry.context import approximate_tokens

packed = pack_documents(documents, 1000, approximate_tokens)
print(packed.tokens, packed.dropped, packed.truncated)
recent = window_history(messages, max_turns=3, max_tokens=None, count=approximate_tokens)
```
