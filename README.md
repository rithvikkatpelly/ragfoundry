# ragfoundry

[![PyPI](https://img.shields.io/pypi/v/ragfoundry)](https://pypi.org/project/ragfoundry/)
[![Python](https://img.shields.io/pypi/pyversions/ragfoundry)](https://pypi.org/project/ragfoundry/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Production RAG pipelines from a single configuration. ragfoundry wires LangChain components into
LangGraph workflows, covering everything from document loading to chat, evaluation and tracing, so
you choose the techniques instead of writing the plumbing.

```python
from ragfoundry import RAGConfig, RAGPipeline

rag = RAGPipeline(RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma"},
    "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
}))
rag.ingest("./docs")
print(rag.query("What is the refund policy?").answer)
```

| Stage | Options |
|---|---|
| Sources | PDF, DOCX, TXT, Markdown, CSV, XLSX/XLS, web pages (with optional same-site crawling) |
| Chunking | `fixed`, `recursive`, `semantic`, `llm`; size in characters or tokens; custom strategies |
| Embeddings | Hugging Face, OpenAI, Gemini |
| Vector stores | Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, PGVector |
| LLMs | Groq, OpenAI, Gemini, Claude, Ollama |
| Retrieval | Hybrid BM25 + vector search, reranking, query rewrite / multi-query / HyDE, corrective retrieval loop, metadata filters |
| Chat | Conversation sessions, follow-up question rewriting, history window |
| Context window | Token budget for retrieved passages and for chat history |
| Evaluation | Retrieval: hit rate, precision, recall, MRR, MAP, nDCG. Generation: faithfulness, answer relevance, answer correctness, context relevance, context recall. Optional RAGAS backend |
| Observability | LangSmith or Langfuse |

## Contents

1. [Installation](#1-installation)
2. [API keys and local models](#2-api-keys-and-local-models)
3. [Quickstart](#3-quickstart)
4. [Configuration: Python or YAML](#4-configuration-python-or-yaml)
5. [Loading documents](#5-loading-documents)
6. [Chunking](#6-chunking)
7. [Embeddings](#7-embeddings)
8. [Vector stores](#8-vector-stores)
9. [LLMs](#9-llms)
10. [Asking questions](#10-asking-questions)
11. [Chat and conversation history](#11-chat-and-conversation-history)
12. [Context window limits](#12-context-window-limits)
13. [Improving retrieval](#13-improving-retrieval)
14. [Generation settings](#14-generation-settings)
15. [Keeping the index up to date](#15-keeping-the-index-up-to-date)
16. [Evaluation](#16-evaluation)
17. [Tracing with LangSmith or Langfuse](#17-tracing-with-langsmith-or-langfuse)
18. [Using ragfoundry in an application](#18-using-ragfoundry-in-an-application)
19. [Extending ragfoundry](#19-extending-ragfoundry)
20. [Examples](#20-examples)
21. [Configuration reference](#21-configuration-reference)
22. [How a query runs](#22-how-a-query-runs)
23. [API reference](#23-api-reference)
24. [Known limitations](#24-known-limitations)
25. [Troubleshooting](#25-troubleshooting)
26. [Development and releases](#26-development-and-releases)

---

## 1. Installation

ragfoundry needs Python 3.10 or newer. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

The core package is small. Each provider is an optional extra, so you install only what you use.
Pick one item from each row:

| You need | Extra |
|---|---|
| Document types | `loaders` (all of PDF, DOCX, Excel, web, token counting) or individually `pdf`, `docx`, `excel`, `web`, `tokens`. TXT, MD and CSV need nothing. |
| Embeddings | `huggingface`, `openai` or `gemini` |
| Vector store | `chroma`, `faiss`, `pinecone`, `qdrant`, `weaviate`, `milvus` or `pgvector` |
| LLM | `groq`, `openai`, `gemini`, `claude` or `ollama` |
| Reranking (optional) | `huggingface` for a local cross-encoder, `cohere` for Cohere Rerank |
| Exact token counting (optional) | `tokens` |
| Evaluation with RAGAS (optional) | `ragas` |
| Tracing (optional) | `langsmith` or `langfuse` |

```bash
# Local embeddings, Chroma and Groq, reading PDFs
pip install "ragfoundry[pdf,huggingface,chroma,groq]"

# Fully local: Hugging Face embeddings, Chroma, Ollama
pip install "ragfoundry[loaders,huggingface,chroma,ollama]"

# Hosted: OpenAI embeddings and LLM, Qdrant, Langfuse tracing
pip install "ragfoundry[loaders,openai,qdrant,langfuse]"

# Everything
pip install "ragfoundry[all]"
```

If you select a provider whose extra is missing, the error tells you the exact `pip install`
command to run.

## 2. API keys and local models

**Hosted providers** need an API key. Export it as an environment variable; each SDK reads its own
variable, so keys don't have to be in your code or config.

| Provider | Environment variable | Where to get a key |
|---|---|---|
| Groq | `GROQ_API_KEY` | console.groq.com |
| OpenAI (embeddings and LLM) | `OPENAI_API_KEY` | platform.openai.com |
| Gemini (embeddings and LLM) | `GOOGLE_API_KEY` | aistudio.google.com |
| Claude | `ANTHROPIC_API_KEY` | console.anthropic.com |
| Pinecone | `PINECONE_API_KEY` | app.pinecone.io |
| Cohere Rerank | `COHERE_API_KEY` | dashboard.cohere.com |

```bash
export GROQ_API_KEY="gsk_..."
```

You can also pass a key in the config (`"api_key": ...`), which is useful in notebooks:

```python
from google.colab import userdata          # in Google Colab
llm = {"provider": "groq", "model": "llama-3.3-70b-versatile",
       "api_key": userdata.get("GROQ_API_KEY")}
```

Keep keys in a `.env` file or a secret manager, never in source control.

**Local providers** need no key:

- **Hugging Face** models download on first use, for example
  `sentence-transformers/all-MiniLM-L6-v2`. Set `HF_TOKEN` only for gated models.
- **Ollama**: install it from ollama.com, pull a model, and keep the server running:

  ```bash
  ollama pull llama3.2
  ollama serve        # skip if the Ollama app is already running
  ```

Model names change often. The ones in this README are examples; use any model your provider
currently offers.

## 3. Quickstart

Put a few documents in a `docs/` folder, then:

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "my_docs"},
    "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temperature": 0},
})

with RAGPipeline(config) as rag:
    report = rag.ingest("./docs")
    print(report.to_dict())        # documents_loaded, chunks_added, ...

    result = rag.query("What is the refund policy?")
    print(result.answer)           # cites passages as [1], [2] ...
    print(result.sources)          # files or URLs used, best first

    chat = rag.chat()              # a conversation with memory
    chat.ask("Who approves refunds?")
    print(chat.ask("And how long does it take?").answer)
```

Run the pipeline as a `with` block, or call `rag.close()` when you are done, so connections close
and traces are flushed.

## 4. Configuration: Python or YAML

Every setting lives in one `RAGConfig`. Only three sections are needed: `embeddings`,
`vector_store` and, for answers, `llm`. Everything else has a sensible default. Unknown keys are
rejected, so a typo such as `temprature` fails at once instead of being ignored.

**In Python**, pass a dict:

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
    "vector_store": {"provider": "qdrant", "url": "http://localhost:6333", "collection_name": "kb"},
    "llm": {"provider": "claude", "model": "claude-sonnet-5-5", "max_tokens": 1024},
    "retrieval": {"top_k": 5, "hybrid": {"enabled": True}},
})
rag = RAGPipeline(config)
```

**In YAML**, write the same structure to a file:

```yaml
# rag.yaml
embeddings:
  provider: openai
  model: text-embedding-3-small

vector_store:
  provider: qdrant
  url: ${QDRANT_URL:-http://localhost:6333}
  collection_name: kb

llm:
  provider: claude
  model: claude-sonnet-5-5
  max_tokens: 1024
```

```python
rag = RAGPipeline.from_yaml("rag.yaml")
```

In YAML, `${VAR}` reads an environment variable and `${VAR:-default}` adds a fallback. A variable
that is unset and has no default stops with an error rather than becoming an empty string.
[`examples/rag.yaml`](examples/rag.yaml) lists every section with comments, and
[section 21](#21-configuration-reference) lists every setting.

## 5. Loading documents

`rag.ingest()` accepts a file, a folder, a URL, or a list of any of these. Folders are searched
recursively for supported files.

```python
rag.ingest("./docs")
rag.ingest("./contracts/master.pdf")
rag.ingest(["./policies", "./faq.md", "https://example.com/handbook"])
```

| Type | Extensions | Extra | Metadata added |
|---|---|---|---|
| PDF | `.pdf` | `pdf` | `page` |
| Word | `.docx` | `docx` | |
| Text | `.txt` | none | |
| Markdown | `.md`, `.markdown` | none | |
| CSV | `.csv` | none | `row`; each row becomes a `column: value` passage |
| Excel | `.xlsx`, `.xlsm`, `.xls` | `excel` | `sheet`, `row` |
| Web page | `http://`, `https://` | `web` | `title` |

Every chunk also gets `source` (the absolute file path, or the URL), `file_type`, `start_index`
and a stable `chunk_id`.

**Choosing files in a folder.** `loading.glob` controls which files are taken, for example
`"**/*.pdf"` for PDFs only.

**Unreadable files.** By default one bad file stops the ingest. With `on_error: skip` it is
skipped and listed in the report:

```python
config = RAGConfig.model_validate({..., "loading": {"on_error": "skip"}})
report = rag.ingest("./docs")
for failure in report.failures:
    print(failure.source, failure.error)
```

**Crawling a website.** A URL fetches only that page unless you set a link depth:

```python
"loading": {
    "web": {
        "max_depth": 2,           # follow links two levels deep
        "max_pages": 100,
        "same_domain": True,      # stay on the starting site
        "respect_robots": True,   # obey robots.txt
        "delay_seconds": 0.5,     # pause between requests
    }
}
```

**Loading without indexing.** To inspect what a loader produces:

```python
from ragfoundry import load_sources
from ragfoundry.config import LoadingConfig

result = load_sources(["./docs"], LoadingConfig())
print(len(result.documents), result.failures)
```

## 6. Chunking

Choose a strategy before your first ingest. `recursive` is a good default.

```python
"chunking": {
    "strategy": "recursive",      # fixed | recursive | semantic | llm
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "length_unit": "characters",  # or "tokens" (needs the "tokens" extra)
}
```

| Strategy | How it splits | Cost |
|---|---|---|
| `fixed` | Exact windows of `chunk_size`, overlapping by `chunk_overlap`. Ignores structure. | None |
| `recursive` | Paragraphs, then lines, then words, keeping each piece under `chunk_size`. Set `separators` to change the order. | None |
| `semantic` | Embeds each sentence with `buffer_size` neighbours and breaks where similarity drops past `semantic.breakpoint_percentile`. | Embedding calls at ingest |
| `llm` | The LLM picks topic boundaries between numbered sentences, one call per `llm.window_chars` of text. A reply it can't parse falls back to recursive splitting for that window, with a warning. | LLM calls at ingest |

`chunk_size` is an upper bound for every strategy. With `length_unit: tokens`, sizes are counted
with the tiktoken encoding in `encoding_name`.

To try a strategy without a vector store:

```python
from langchain_core.documents import Document
from ragfoundry import chunk_documents
from ragfoundry.config import ChunkingConfig

chunks = chunk_documents(
    [Document(page_content=open("doc.md").read(), metadata={"source": "doc.md"})],
    ChunkingConfig(strategy="fixed", chunk_size=300, chunk_overlap=50),
)
```

## 7. Embeddings

```python
# Local, free
"embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}

# OpenAI, or any OpenAI-compatible server through base_url
"embeddings": {"provider": "openai", "model": "text-embedding-3-small"}

# Gemini
"embeddings": {"provider": "gemini", "model": "models/text-embedding-004"}
```

`options` is passed to the LangChain embeddings class, for example
`{"model_kwargs": {"device": "cuda"}}` for Hugging Face on a GPU.

A collection remembers which embedding model filled it. Switching models on an existing
collection raises an error, because mixing them silently breaks search; use a new
`collection_name` instead.

## 8. Vector stores

Choose one with `vector_store.provider`. `collection_name` (default `ragfoundry`) separates
datasets in the same store.

**No server needed.** Data is kept under `persist_dir` (default `.ragfoundry/`):

```python
"vector_store": {"provider": "chroma", "collection_name": "kb"}
"vector_store": {"provider": "faiss", "collection_name": "kb"}
"vector_store": {"provider": "qdrant", "path": "./qdrant_data"}   # Qdrant local mode
"vector_store": {"provider": "milvus"}                             # Milvus Lite file
```

**Servers:**

```python
"vector_store": {"provider": "chroma", "host": "localhost", "port": 8000}
"vector_store": {"provider": "qdrant", "url": "http://localhost:6333", "api_key": None}
"vector_store": {"provider": "weaviate", "url": "http://localhost:8080", "grpc_port": 50051}
"vector_store": {"provider": "milvus", "uri": "http://localhost:19530", "token": None}
"vector_store": {"provider": "pgvector",
                 "connection": "postgresql+psycopg://rag:rag@localhost:5432/rag"}
"vector_store": {"provider": "pinecone", "index_name": "my-index"}  # collection_name = namespace
```

Start a server with Docker:

```bash
docker run -d -p 6333:6333 qdrant/qdrant                                    # Qdrant

docker run -d -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:1.30.0                           # Weaviate

docker run -d -p 5432:5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag pgvector/pgvector:pg16                                 # PGVector
```

For a Milvus server, follow the Milvus standalone install guide.

**Pinecone** needs its index created in the Pinecone console first, with the cosine metric and
the same dimension as your embedding model (384 for `all-MiniLM-L6-v2`, 1536 for OpenAI
`text-embedding-3-small`). The other stores create their collection on first use.

## 9. LLMs

```python
"llm": {"provider": "groq",   "model": "llama-3.3-70b-versatile"}
"llm": {"provider": "openai", "model": "gpt-4o-mini"}
"llm": {"provider": "gemini", "model": "gemini-2.5-flash"}
"llm": {"provider": "claude", "model": "claude-sonnet-5-5", "max_tokens": 1024}
"llm": {"provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434"}
```

| Setting | Meaning |
|---|---|
| `temperature` | Sent only when set; some models reject it. `0` gives the most repeatable answers. |
| `max_tokens` | Maximum answer length. |
| `api_key` | Overrides the provider's environment variable. |
| `base_url` | OpenAI-compatible servers, a Claude proxy, or a remote Ollama. |
| `options` | Passed to the LangChain chat model class, for example `{"reasoning": False}` for Ollama reasoning models. |

The LLM is created the first time it's needed, so ingest-only scripts and plain retrieval don't
need an `llm` section at all.

## 10. Asking questions

```python
result = rag.query("What is the refund policy?")
```

`query` returns a `RAGResult`:

| Field | Contents |
|---|---|
| `answer` | The generated answer, citing passages as `[1]`, `[2]` ... |
| `sources` | Distinct sources of the passages, best first |
| `documents` | The passages sent to the LLM, as LangChain `Document`s with metadata |
| `queries` | Every search query used (more than one with query transforms or retries) |
| `retries` | Rounds of the corrective loop |
| `abstained` | `True` when nothing relevant was found and the abstain message was returned |
| `standalone_question` | The rewritten question used for search, when chat history changed it |
| `context_tokens` | Tokens of retrieved context sent to the LLM |

`result.to_dict()` gives all of it as JSON-ready data.

**Options on every query call:**

```python
# Only search chunks whose metadata matches exactly
rag.query("Summarise the policy", where={"file_type": "pdf"})
rag.query("What changed?", where={"source": "/abs/path/to/policy.pdf", "page": 3})

# Use more or fewer passages than retrieval.top_k
rag.query("Give me every detail", top_k=10)
```

`source` values are absolute paths (or URLs); `rag.sources()` lists them.

**Streaming** yields the answer as it is generated:

```python
for token in rag.stream("Summarise the security section"):
    print(token, end="", flush=True)
```

**Retrieval only**, without calling the LLM:

```python
for doc in rag.retrieve("refund timelines", top_k=3):
    print(doc.metadata["source"], doc.page_content[:80])
```

**Async** versions for web servers: `await rag.aquery(...)` and `async for token in rag.astream(...)`.

If no relevant passages are found, the answer is `generation.abstain_message`, `abstained` is
`True`, and the LLM is not asked to answer from nothing.

## 11. Chat and conversation history

`rag.chat()` starts a conversation that remembers what was said. Each new question is answered with
the recent conversation in view, so follow-ups like "how long does that take?" work.

```python
chat = rag.chat()

chat.ask("What does the refund policy say?")
result = chat.ask("And how long does it take?")

print(result.standalone_question)   # "How long does a refund take under the refund policy?"
print(result.answer)
```

Two things happen for each follow-up:

1. **Question rewriting.** The LLM rewrites the follow-up into a standalone question
   (`result.standalone_question`), and that is what gets searched. Searching for "and how long
   does it take?" alone would find nothing useful. Turn it off with
   `context_window.condense_question: false` to save one LLM call per question.
2. **History in the prompt.** The most recent turns are sent to the LLM along with the retrieved
   passages, so the answer can refer back to the conversation.

`ChatSession` methods:

```python
chat.ask(question, where=None, top_k=None)          # -> RAGResult
chat.stream(question)                                # yields tokens; turn recorded when finished
await chat.aask(question)
async for token in chat.astream(question): ...
len(chat)                                            # completed turns
chat.history                                         # list of LangChain messages
chat.reset()                                         # forget everything
```

**Saving and resuming a conversation.** Sessions live in memory. To keep a conversation across
restarts or server processes, store `to_dicts()` and pass it back:

```python
import json

saved = chat.to_dicts()        # [{"role": "user", "content": ...}, {"role": "assistant", ...}]
json.dump(saved, open("chat.json", "w"))

chat = rag.chat(json.load(open("chat.json")))
```

**Managing history yourself.** `query`, `stream` and their async versions accept `history`
directly, as LangChain messages, `{"role", "content"}` dicts or `(role, content)` tuples. Roles
can be `user`/`human` and `assistant`/`ai`:

```python
history = [
    {"role": "user", "content": "What does the refund policy say?"},
    {"role": "assistant", "content": "Refunds are issued within five business days [1]."},
]
result = rag.query("Who approves them?", history=history)
```

How much history is sent is controlled by `context_window`, described next.

## 12. Context window limits

Every LLM has a maximum context size, and long prompts cost more and can lower answer quality.
`context_window` limits what is put into each prompt:

```python
"context_window": {
    "max_context_tokens": 4000,      # retrieved passages; unset = send all top_k passages
    "truncate_last_passage": True,   # shorten the passage that crosses the budget
    "max_history_turns": 5,          # recent chat turns sent with each question
    "max_history_tokens": 2000,      # also cap history by tokens; unset = no token cap
    "condense_question": True,       # rewrite follow-ups before searching
    "token_counter": "approximate",  # approximate | tiktoken
}
```

**Passage budget.** Passages are added in rank order until `max_context_tokens` is reached, so
the most relevant ones always make it in. The passage that would cross the limit is shortened to
fill the remaining space, ending in `…` and marked with `metadata["truncated"] = True`. Set
`truncate_last_passage: false` to drop it instead. Passages after it are left out.
`result.context_tokens` reports what was actually sent, and `result.documents` holds exactly the
passages the LLM saw.

**History window.** Only the last `max_history_turns` questions, each with its answer, are sent. If
`max_history_tokens` is set, the oldest turns are dropped until the rest fit. Whole turns are
always kept together, so the model never sees an answer without its question. `ChatSession` keeps
the full history; the window only affects what each request sends. `max_history_turns: 0` sends no
history and skips question rewriting.

**Counting tokens.**

- `approximate` (default) counts about four characters per token, needs nothing extra, and is
  close enough for budgeting with most models.
- `tiktoken` counts exactly with the OpenAI encoding in `encoding_name`. It needs the `tokens`
  extra.
- Any function can be used, for example your model's own tokenizer:

  ```python
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
  rag = RAGPipeline(config, token_counter=lambda text: len(tokenizer.encode(text)))
  ```

**Choosing budgets.** Leave room for the system prompt, the question and the answer
(`llm.max_tokens`). For an 8,000-token model, `max_context_tokens: 4000` with
`max_history_tokens: 1500` is a reasonable start. Watch `result.context_tokens` and the
`truncated` flag while you tune.

Both functions are available for your own code:

```python
from ragfoundry import pack_documents, window_history
from ragfoundry.context import approximate_tokens

packed = pack_documents(docs, 1000, approximate_tokens)    # .documents, .tokens, .dropped
recent = window_history(messages, 3, None, approximate_tokens)
```

## 13. Improving retrieval

Turn these on one at a time, and measure each change with [evaluation](#16-evaluation).

```python
"retrieval": {
    "top_k": 5,        # passages given to the LLM
    "fetch_k": 30,     # candidates gathered before fusion and reranking

    # Keyword + vector search; helps with names, codes and part numbers
    "hybrid": {"enabled": True, "vector_weight": 1.0, "bm25_weight": 1.0},

    # Re-score candidates with a stronger model
    "rerank": {"provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},

    # Rephrase the question before searching; each strategy adds an LLM call
    "query_transform": {"strategies": ["multi_query"], "num_queries": 3},

    # Grade passages with the LLM; retry with a new query when too few are relevant
    "corrective": {"enabled": True, "max_retries": 1, "min_relevant_docs": 1},
}
```

| Technique | What it does | Cost |
|---|---|---|
| **Hybrid search** | Runs BM25 keyword search alongside vector search and merges the rankings with reciprocal rank fusion (`rrf_k`). The weights tilt the merge. | Small, in memory |
| **Reranking** | Re-scores `fetch_k` candidates against the question and keeps `top_k`. `huggingface` runs a cross-encoder locally; `cohere` uses Cohere Rerank (`cohere` extra, `COHERE_API_KEY`). | One model pass |
| **`rewrite`** | Replaces the question with a clearer search query. | 1 LLM call |
| **`multi_query`** | Adds `num_queries` alternative phrasings and searches them all. | 1 LLM call |
| **`hyde`** | Writes a hypothetical answer passage and uses it for vector search only. | 1 LLM call |
| **Corrective loop** | Asks the LLM whether each passage is useful. If fewer than `min_relevant_docs` pass, it writes a new query and searches again, up to `max_retries` times. Irrelevant passages are removed before generation. | 1 LLM call per passage, per round |

A sensible order is hybrid, then reranking, then query transforms, then the corrective loop. The
first two add little latency; the last two add LLM calls to every query.

## 14. Generation settings

```python
"generation": {
    "system_prompt": "You are a support assistant for Acme. Answer in two sentences. Cite [n].",
    "abstain_message": "Sorry, I couldn't find that in our documentation.",
}
```

The default system prompt tells the model to answer only from the numbered passages, cite them as
`[n]`, say so when the answer isn't there, and treat passage text as data rather than
instructions. A custom `system_prompt` replaces it completely, so keep those rules in yours.

## 15. Keeping the index up to date

Run the same ingest again whenever documents change:

```python
report = rag.ingest("./docs")
print(report.chunks_added, report.chunks_unchanged, report.chunks_removed)
```

Only new or edited chunks are embedded, and chunks that disappeared from a re-read file are
removed. If a run is interrupted, run it again and it carries on.

```python
rag.sources()                                # {"/abs/path/policy.pdf": 12, ...}
rag.delete_source("./docs/old-policy.pdf")   # returns the number of chunks removed
rag.delete_source("https://example.com/old-page")
```

`delete_source` accepts relative or absolute paths, as `ingest` does; URLs must match exactly.

Every process that serves the same collection must use the same `persist_dir`. It holds the chunk
record that hybrid search and incremental ingestion rely on.

## 16. Evaluation

Write a test set as JSON Lines, one question per line, labelling what should be found and,
optionally, a reference answer:

```json
{"question": "How long do refunds take?", "reference_answer": "Five business days.", "relevant_sources": ["refunds.md"]}
{"question": "Who approves travel?", "reference_contexts": ["Travel must be approved by a line manager"]}
```

| Field | Meaning |
|---|---|
| `question` | Required |
| `relevant_sources` | File names, path endings or URLs that should be retrieved |
| `reference_contexts` | Passages a correct result should overlap |
| `relevant_ids` | Exact `chunk_id` values |
| `reference_answer` | Needed for `answer_correctness` and `context_recall` |

Retrieval metrics need at least one of the three relevance fields. A `.json` file holding a list
works too, as does a list of `EvalSample` objects.

```python
report = rag.evaluate("eval.jsonl", mode="retrieval")   # fast, no LLM judging
report = rag.evaluate("eval.jsonl", mode="generation")  # LLM-judged answer quality
report = rag.evaluate("eval.jsonl", mode="all")         # both (default)

for metric, stats in report.summary.items():
    print(f"{metric:<20} {stats['mean']:.3f}  (n={stats['count']})")
report.save("report.json")                              # per-question rows and errors
```

```python
from ragfoundry import EvalSample

samples = [EvalSample(question="How long do refunds take?",
                      reference_answer="Five business days.",
                      relevant_sources=["refunds.md"])]
report = rag.evaluate(samples)
```

**Retrieval metrics**, each at every `k` in `evaluation.k_values`: `hit_rate@k`, `precision@k`,
`recall@k`, `ndcg@k`, plus `mrr` and `map`. When several chunks come from one relevant document,
each counts towards precision, but the document counts once towards recall, MAP and nDCG.

**Generation metrics**, judged by an LLM:

| Metric | Measures | Needs |
|---|---|---|
| `faithfulness` | Share of the answer's claims supported by the retrieved passages | retrieved passages |
| `answer_relevance` | How directly the answer addresses the question | |
| `context_relevance` | Share of retrieved passages useful for the question | retrieved passages |
| `answer_correctness` | Agreement with the reference answer | `reference_answer` |
| `context_recall` | Share of the reference answer's claims found in the passages | `reference_answer` |

`count` is the number of samples actually scored; a metric is skipped where it doesn't apply.

**Judge model.** The judge defaults to the pipeline LLM. A different model gives fairer scores,
because it isn't grading its own answers:

```python
"evaluation": {
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},
    "k_values": [1, 3, 5],
    "max_concurrency": 4,      # questions evaluated in parallel
}
```

**RAGAS.** Install the `ragas` extra and pass `backend="ragas"` to score generation with RAGAS
metrics instead of the built-in ones.

The metric functions also work on their own:

```python
from ragfoundry.evaluation import recall_at_k, ndcg_at_k

recall_at_k(["a", "x", "b"], {"a", "b"}, k=3)    # 1.0
```

## 17. Tracing with LangSmith or Langfuse

```python
# LangSmith (langsmith extra)
"observability": {"provider": "langsmith", "api_key": "...", "project": "my-rag"}

# Langfuse (langfuse extra)
"observability": {
    "provider": "langfuse",
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-...",
    "endpoint": "https://cloud.langfuse.com",   # or your self-hosted URL
}
```

In YAML, read keys from the environment, for example `api_key: ${LANGSMITH_API_KEY}`.

Each ingest, query, chat turn and evaluation appears as one trace (`ragfoundry.query` and so on),
with every graph step and LLM call nested inside. `tags` adds labels to every trace. Traces are
sent in the background, so close the pipeline (or call `rag.flush()`) before your program exits.

## 18. Using ragfoundry in an application

Create one pipeline when the application starts and share it; it holds model clients and
connections. Create one `ChatSession` per conversation.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from ragfoundry import ChatSession, RAGPipeline

rag = RAGPipeline.from_yaml("rag.yaml")
sessions: dict[str, ChatSession] = {}       # use Redis or a database across processes


@asynccontextmanager
async def lifespan(app):
    yield
    rag.close()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    session = sessions.setdefault(request.session_id, rag.chat())
    result = await session.aask(request.question)
    return {"answer": result.answer, "sources": result.sources}
```

To share conversations between several server processes, store `session.to_dicts()` in your
database after each turn and rebuild with `rag.chat(saved)`, or call
`rag.aquery(question, history=saved)` directly.

## 19. Extending ragfoundry

**Bring your own components.** Any LangChain object replaces the configured one:

```python
rag = RAGPipeline(
    config,
    llm=my_chat_model,               # any BaseChatModel
    embeddings=my_embeddings,        # any Embeddings
    vector_store=my_vector_store,    # any VectorStore
    reranker=my_reranker,            # any object with rerank(query, documents, top_n)
    token_counter=my_count_function, # any str -> int function
)
```

**Add a file type:**

```python
from ragfoundry import register_loader

@register_loader(".pptx")
def pptx_loader(path, loading_config):
    return MyPptxLoader(path)        # any LangChain BaseLoader
```

**Add a chunking strategy:**

```python
from ragfoundry import register_chunker

@register_chunker("by_heading")
def heading_chunker(chunking_config, resources):
    return MyHeadingSplitter(chunk_size=chunking_config.chunk_size)   # any TextSplitter
```

Then set `"chunking": {"strategy": "by_heading"}`. `resources.get_embeddings()` and
`resources.get_llm()` give your splitter the pipeline's models.

## 20. Examples

The [`examples/`](examples/) folder has runnable scripts using the sample documents in
[`examples/sample_docs/`](examples/sample_docs/):

| File | Shows |
|---|---|
| [`01_quickstart.py`](examples/01_quickstart.py) | Smallest complete pipeline, fully local with Ollama |
| [`02_groq_pdf_chat.py`](examples/02_groq_pdf_chat.py) | Chat with a PDF in the terminal using Groq, with streaming and follow-ups |
| [`03_advanced_retrieval.py`](examples/03_advanced_retrieval.py) | Hybrid search, reranking, multi-query, corrective loop, context budget, metadata filters |
| [`04_evaluation.py`](examples/04_evaluation.py) | Retrieval and generation metrics from a file and from code |
| [`05_fastapi_chat.py`](examples/05_fastapi_chat.py) | A chat API with one conversation per session |
| [`06_yaml_config.py`](examples/06_yaml_config.py) | Loading `rag.yaml`, chat, and removing a source |
| [`rag.yaml`](examples/rag.yaml) | Every configuration section, commented |
| [`eval.jsonl`](examples/eval.jsonl) | A labelled test set for the sample documents |
| [`colab_quickstart.ipynb`](examples/colab_quickstart.ipynb) | The full tour in Google Colab |

Run them from the repository root, for example `python examples/02_groq_pdf_chat.py report.pdf`.

**Recipe: chat with a PDF in one Colab cell**

```python
%pip install -q "ragfoundry[huggingface,chroma,groq,pdf]"
from google.colab import userdata
from ragfoundry import RAGConfig, RAGPipeline

rag = RAGPipeline(RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "my_pdf"},
    "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile",
            "api_key": userdata.get("GROQ_API_KEY"), "temperature": 0},
    "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
    "context_window": {"max_context_tokens": 3000},
}))
rag.ingest("/content/your_file.pdf")
chat = rag.chat()
while question := input("You: ").strip():
    print("Assistant:", chat.ask(question).answer)
```

**Recipe: answer only from one document**

```python
from pathlib import Path

policy = str(Path("docs/policy.pdf").resolve())
rag.query("What is the notice period?", where={"source": policy})
```

**Recipe: show citations with page numbers**

```python
result = rag.query("What is the notice period?")
print(result.answer)
for number, doc in enumerate(result.documents, start=1):
    print(f"[{number}] {Path(doc.metadata['source']).name}, page {doc.metadata.get('page', '-')}")
```

**Recipe: compare two configurations**

```python
base = {"embeddings": {...}, "vector_store": {"provider": "chroma", "collection_name": "kb"},
        "llm": {...}}
for name, retrieval in {"vector": {}, "hybrid": {"hybrid": {"enabled": True}}}.items():
    with RAGPipeline(RAGConfig.model_validate({**base, "retrieval": retrieval})) as rag:
        rag.ingest("./docs")
        report = rag.evaluate("eval.jsonl", mode="retrieval")
        print(name, round(report.summary["recall@5"]["mean"], 3))
```

## 21. Configuration reference

All settings with their defaults.

```yaml
persist_dir: .ragfoundry       # local state: chunk record, and Chroma/FAISS/Qdrant/Milvus Lite data
batch_size: 64                 # chunks written to the vector store per batch

loading:
  on_error: raise              # raise | skip
  encoding: utf-8
  glob: "**/*"                 # which files to take from a folder
  web:
    max_depth: 0               # 0 fetches only the given URL
    max_pages: 50
    same_domain: true
    respect_robots: true
    delay_seconds: 0.5
    timeout_seconds: 30
    max_bytes: 10000000
    user_agent: ragfoundry/0.2

chunking:
  strategy: recursive          # fixed | recursive | semantic | llm | a registered name
  chunk_size: 1000
  chunk_overlap: 200
  length_unit: characters      # characters | tokens
  encoding_name: cl100k_base   # tokenizer used when length_unit is tokens
  separators: null             # custom separators for recursive
  semantic:
    breakpoint_percentile: 95
    buffer_size: 1
  llm:
    window_chars: 8000

embeddings:                    # required for ingest and search
  provider: huggingface        # huggingface | openai | gemini
  model: sentence-transformers/all-MiniLM-L6-v2
  api_key: null                # default: the provider's environment variable
  base_url: null               # OpenAI-compatible endpoints
  options: {}                  # passed to the LangChain embeddings class

vector_store:                  # required; one provider block
  provider: chroma
  collection_name: ragfoundry
  # chroma:   persist_directory, or host and port
  # faiss:    path
  # pinecone: index_name (required), api_key; collection_name is used as the namespace
  # qdrant:   url and api_key, or path for local mode
  # weaviate: url, grpc_port, api_key (cloud)
  # milvus:   uri and token; omit uri for Milvus Lite
  # pgvector: connection (required)

llm:                           # required for answers
  provider: ollama             # groq | openai | gemini | claude | ollama
  model: llama3.2
  api_key: null
  base_url: null
  temperature: null            # sent only when set
  max_tokens: null
  options: {}                  # passed to the LangChain chat model class

retrieval:
  top_k: 5
  fetch_k: 20                  # must be >= top_k
  hybrid: {enabled: false, rrf_k: 60, vector_weight: 1.0, bm25_weight: 1.0}
  rerank: null                 # {provider: huggingface | cohere, model, api_key}
  query_transform: {strategies: [], num_queries: 3}    # rewrite | multi_query | hyde
  corrective: {enabled: false, max_retries: 1, min_relevant_docs: 1}

context_window:
  max_context_tokens: null     # token budget for retrieved passages; null = no limit
  truncate_last_passage: true
  max_history_turns: 5         # 0 disables chat history
  max_history_tokens: null
  condense_question: true
  token_counter: approximate   # approximate | tiktoken
  encoding_name: cl100k_base

generation:
  system_prompt: null          # replaces the default prompt
  abstain_message: "I don't have enough information in the provided documents to answer."

observability:
  provider: none               # none | langsmith | langfuse
  project: ragfoundry
  api_key: null                # LangSmith
  public_key: null             # Langfuse
  secret_key: null             # Langfuse
  endpoint: null
  tags: []

evaluation:
  llm: null                    # judge model; defaults to the pipeline llm
  k_values: [1, 3, 5]
  max_concurrency: 1
```

## 22. How a query runs

The query graph contains only the stages you enable:

```
prepare → [condense] → [transform] → retrieve → [rerank] → [grade ⇄ rewrite] → generate | abstain
```

- **prepare** applies the history window.
- **condense** rewrites a follow-up into a standalone question, when there is chat history.
- **transform** applies `rewrite`, `multi_query` and `hyde`.
- **retrieve** searches the vector store, and BM25 when hybrid is on, merging rankings with
  reciprocal rank fusion.
- **rerank** re-scores candidates and keeps `top_k`.
- **grade / rewrite** is the corrective loop.
- **generate** packs passages into the context budget and sends the system prompt, the history
  window, the passages and the original question to the LLM.
- **abstain** returns the abstain message without calling the LLM when nothing relevant was found.

Search stages use the standalone question; generation uses the user's own wording.

## 23. API reference

| Object | Members |
|---|---|
| `RAGConfig` | `model_validate(dict)`, `from_yaml(path)` |
| `RAGPipeline(config, *, llm, embeddings, vector_store, reranker, token_counter)` | `from_yaml(path, **components)`, `ingest(sources)`, `sources()`, `delete_source(source)`, `query(...)`, `aquery(...)`, `stream(...)`, `astream(...)`, `retrieve(...)`, `chat(history=None)`, `evaluate(dataset, *, mode, backend, k_values)`, `flush()`, `close()`; also a context manager |
| `RAGResult` | `question`, `answer`, `documents`, `sources`, `queries`, `retries`, `abstained`, `standalone_question`, `context_tokens`, `to_dict()` |
| `ChatSession` | `ask`, `aask`, `stream`, `astream`, `history`, `reset()`, `to_dicts()`, `len()` |
| `IngestReport` | `documents_loaded`, `chunks_total`, `chunks_added`, `chunks_unchanged`, `chunks_removed`, `failures`, `to_dict()` |
| `EvalSample` | `question`, `reference_answer`, `relevant_sources`, `reference_contexts`, `relevant_ids` |
| `EvalReport` | `summary`, `rows`, `to_dict()`, `save(path)` |
| Functions | `load_sources`, `chunk_documents`, `load_dataset`, `pack_documents`, `window_history`, `register_loader`, `register_chunker` |
| Errors | `RAGFoundryError` (base), `ConfigurationError`, `IngestionError`, `MissingDependencyError` |

`query`, `aquery`, `stream`, `astream` and `retrieve` all take
`(question, *, where=None, top_k=None, history=None)`.

## 24. Known limitations

- **macOS with FAISS and Hugging Face models**: the `faiss-cpu` and `torch` wheels each bundle an
  OpenMP runtime, and using both in one process can abort it (`OMP: Error #15`). ragfoundry warns
  about this combination. Use Chroma or Qdrant with Hugging Face models on macOS.
- **Legacy `.doc` files** are not supported; convert them to `.docx`.
- **Scanned PDFs** need OCR first. A PDF with no extractable text is reported as a failure.
- **BM25** is held in memory and scores every chunk on each query, which suits collections up to a
  few hundred thousand chunks.
- **Web crawling** stays within one site by default and respects `robots.txt`. Pages that vanish
  from a site stay indexed until you remove them with `delete_source`.
- **Chat sessions** are kept in memory; persist `to_dicts()` yourself to share them across
  processes.
- **The approximate token counter** can be off by 20% or more for code, non-English text or
  unusual tokenizers. Use `tiktoken` or your model's tokenizer when budgets are tight.
- **RAGAS 0.4** imports a module that current `langchain-community` has removed. ragfoundry
  provides a placeholder for that module, and only when it is genuinely absent.

## 25. Troubleshooting

| Problem | Fix |
|---|---|
| `'X' is required for this feature. Install it with: pip install 'ragfoundry[...]'` | Run the command shown. |
| `Environment variable 'X' is referenced but not set` | Export the variable, or give a default: `${X:-value}`. |
| `No LLM configured` | Add an `llm` section, or pass `llm=...`. Retrieval alone needs no LLM. |
| `This collection was indexed with embeddings '...'` | You changed the embedding model. Use a new `collection_name`, or delete `persist_dir` and the old collection, then ingest again. |
| `No extractable text in file.pdf` | The PDF is scanned. Run OCR on it first. |
| `robots.txt disallows fetching ...` | The site forbids crawlers. Set `loading.web.respect_robots: false` only if you have permission. |
| Model not found or decommissioned | Model names change; pick a current one from your provider's model list. |
| Answers are always the abstain message | Check `rag.sources()` to confirm documents are indexed. With `corrective` on, the grader may be too strict for a small model. |
| A `where` filter returns nothing | Values must match exactly. `source` is an absolute path; copy it from `rag.sources()`. |
| Follow-up questions retrieve the wrong documents | Check `result.standalone_question`. A small LLM may rewrite poorly; try a stronger model, or raise `max_history_turns`. |
| Answers miss details that are in the documents | `max_context_tokens` may be too small. Check `result.context_tokens` and whether the last passage has `truncated` set. |
| `OMP: Error #15` on macOS | See the FAISS limitation above. |

## 26. Development and releases

```bash
git clone https://github.com/rithvikkatpelly/ragfoundry && cd ragfoundry
uv venv && uv pip install -e ".[all,dev]"
pytest                    # unit tests, no network needed
pytest -m integration     # real services; see tests/test_integration.py for setup
ruff check src tests && mypy src
```

**Releasing a new version:**

1. Raise `version` in `pyproject.toml` and `__version__` in `src/ragfoundry/__init__.py`, and add
   an entry to [CHANGELOG.md](CHANGELOG.md). PyPI never accepts the same version twice.
2. Build and check:

   ```bash
   rm -rf dist && uv build
   uvx twine check dist/*
   ```

3. Upload with a PyPI API token scoped to this project, passed through the environment so it
   isn't saved in your shell history or files:

   ```bash
   read -rs UV_PUBLISH_TOKEN && export UV_PUBLISH_TOKEN   # paste the pypi-... token
   uv publish
   unset UV_PUBLISH_TOKEN
   ```

4. Verify in a clean environment: `pip install "ragfoundry==<version>"`.

## License

MIT. See [LICENSE](LICENSE).
