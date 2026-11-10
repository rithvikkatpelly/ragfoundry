# ragfoundry

Production RAG pipelines from a config file. ragfoundry wires LangChain components into LangGraph
workflows, covering everything from document loading to evaluation and tracing, so you choose the
techniques instead of writing the plumbing.

| Stage | Options |
|---|---|
| Sources | PDF, DOCX, TXT, Markdown, CSV, XLSX/XLS, web pages (with optional same-site crawling) |
| Chunking | `fixed` (sliding window), `recursive`, `semantic`, `llm`; size in characters or tokens |
| Embeddings | Hugging Face, OpenAI, Gemini |
| Vector stores | Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, PGVector |
| LLMs | Groq, OpenAI, Gemini, Claude, Ollama |
| Retrieval | Hybrid BM25 + vector search, reranking, query rewrite / multi-query / HyDE, corrective retrieval loop, metadata filters |
| Evaluation | Retrieval: hit rate, precision, recall, MRR, MAP, nDCG. Generation: faithfulness, answer relevance, answer correctness, context relevance, context recall. Optional RAGAS backend |
| Observability | LangSmith or Langfuse |

**Contents:** [Step-by-step guide](#step-by-step-guide) ·
[Configuration reference](#configuration-reference) · [How a query runs](#how-a-query-runs) ·
[Evaluation details](#evaluation-details) · [Known limitations](#known-limitations) ·
[Troubleshooting](#troubleshooting)

---

## Step-by-step guide

This walks from an empty folder to a tested, traced RAG system. Steps 1 to 7 give you a working
pipeline; steps 8 to 12 make it production ready.

### Step 1: Check your Python version

ragfoundry needs Python 3.10 or newer.

```bash
python --version
```

Create a virtual environment for your project:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2: Install ragfoundry with the extras you need

The core package is small. Each provider is an optional extra, so you install only what you use.
Pick one item from each row:

| You need | Extra |
|---|---|
| Document types | `loaders` (all of PDF, DOCX, Excel, web, token counting) or individually `pdf`, `docx`, `excel`, `web`, `tokens`. TXT, MD and CSV need nothing. |
| Embeddings | `huggingface`, `openai` or `gemini` |
| Vector store | `chroma`, `faiss`, `pinecone`, `qdrant`, `weaviate`, `milvus` or `pgvector` |
| LLM | `groq`, `openai`, `gemini`, `claude` or `ollama` |
| Reranking (optional) | `huggingface` for a local cross-encoder, `cohere` for Cohere Rerank |
| Evaluation with RAGAS (optional) | `ragas` |
| Tracing (optional) | `langsmith` or `langfuse` |

Examples:

```bash
# Fully local: Hugging Face embeddings, Chroma, Ollama
pip install "ragfoundry[loaders,huggingface,chroma,ollama]"

# Hosted: OpenAI embeddings and LLM, Qdrant, Langfuse tracing
pip install "ragfoundry[loaders,openai,qdrant,langfuse]"

# Everything
pip install "ragfoundry[all]"
```

Check the install:

```bash
ragfoundry --help
```

If you later select a provider whose extra is missing, the error message gives the exact
`pip install` command to run.

### Step 3: Get access to your models

**Hosted providers** need an API key. Export it as an environment variable; each SDK reads its own
variable, so you don't need to put keys in the config file.

| Provider | Environment variable | Where to get a key |
|---|---|---|
| OpenAI (embeddings and LLM) | `OPENAI_API_KEY` | platform.openai.com |
| Gemini (embeddings and LLM) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | aistudio.google.com |
| Claude | `ANTHROPIC_API_KEY` | console.anthropic.com |
| Groq | `GROQ_API_KEY` | console.groq.com |
| Pinecone | `PINECONE_API_KEY` | app.pinecone.io |
| Cohere Rerank | `COHERE_API_KEY` | dashboard.cohere.com |

```bash
export OPENAI_API_KEY="sk-..."
```

Keep keys in a `.env` file or your secret manager, never in source control.

**Local providers** need no key:

- **Hugging Face** models download automatically on first use (for example
  `sentence-transformers/all-MiniLM-L6-v2`). Set `HF_TOKEN` only for gated models.
- **Ollama**: install it from ollama.com, then pull a model and keep the server running:

  ```bash
  ollama pull llama3.2
  ollama serve        # skip if the Ollama app is already running
  ```

### Step 4: Start your vector store

These need nothing running. Data is stored in your project's `.ragfoundry/` folder:

- **Chroma** (`provider: chroma`)
- **FAISS** (`provider: faiss`)
- **Qdrant local mode** (`provider: qdrant` with no `url`)
- **Milvus Lite** (`provider: milvus` with no `uri`)

For a server, start one with Docker:

```bash
# Qdrant  ->  url: http://localhost:6333
docker run -d -p 6333:6333 qdrant/qdrant

# Weaviate  ->  url: http://localhost:8080, grpc_port: 50051
docker run -d -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:1.30.0

# PGVector  ->  connection: postgresql+psycopg://rag:rag@localhost:5432/rag
docker run -d -p 5432:5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  pgvector/pgvector:pg16
```

For a Milvus server, follow the Milvus standalone install guide and set `uri` to it, for example
`http://localhost:19530`.

**Pinecone** must have an index created in its console before you ingest. Give the index the same
dimension as your embedding model (384 for `all-MiniLM-L6-v2`, 1536 for OpenAI
`text-embedding-3-small`) and the cosine metric.

### Step 5: Create a config file

```bash
ragfoundry init            # writes rag.yaml in the current folder
```

Open `rag.yaml` and set the three required sections. This example is fully local:

```yaml
embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  provider: chroma
  collection_name: my_docs

llm:
  provider: ollama
  model: llama3.2
```

The same with hosted services:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small

vector_store:
  provider: qdrant
  url: ${QDRANT_URL:-http://localhost:6333}
  collection_name: my_docs

llm:
  provider: claude
  model: claude-sonnet-5-5
  max_tokens: 1024
```

Any value can read the environment with `${VAR}` or `${VAR:-default}`. An unset variable with no
default stops with an error rather than becoming an empty string. Use any model ID your provider
offers; the ones above are examples. The [configuration reference](#configuration-reference) lists
every setting.

### Step 6: Ingest your documents

Pass files, folders or URLs. Folders are searched recursively for supported file types.

```bash
ragfoundry ingest ./docs
ragfoundry ingest ./contracts/master.pdf ./notes https://example.com/handbook
```

Output:

```
Documents: 42  Chunks: 318  Added: 318  Unchanged: 0  Removed: 0
```

To check what is indexed:

```bash
ragfoundry sources
```

Choose a chunking strategy in `rag.yaml` before your first ingest. `recursive` is a good default;
see [Chunking strategies](#chunking-strategies) for the others.

```yaml
chunking:
  strategy: recursive       # fixed | recursive | semantic | llm
  chunk_size: 1000
  chunk_overlap: 200
  length_unit: characters   # or tokens (needs the "tokens" extra)
```

To crawl a website rather than fetch a single page, set the link depth:

```yaml
loading:
  web:
    max_depth: 2
    max_pages: 100
```

By default one unreadable file stops the ingest. To skip it, report it, and continue, set
`loading.on_error: skip`.

### Step 7: Ask questions

From the command line:

```bash
ragfoundry query "What is the refund policy?"
ragfoundry query "What is the refund policy?" --stream    # print tokens as they arrive
ragfoundry query "What is the refund policy?" --json      # answer, sources and passages
```

From Python:

```python
from ragfoundry import RAGPipeline

with RAGPipeline.from_yaml("rag.yaml") as rag:
    result = rag.query("What is the refund policy?")
    print(result.answer)            # cites passages as [1], [2] ...
    print(result.sources)           # source files or URLs, best first
    for doc in result.documents:    # the passages given to the LLM
        print(doc.metadata["source"], doc.page_content[:80])

    # Restrict to documents with matching metadata, and change how many passages are used
    result = rag.query("Summarise the policy", where={"file_type": "pdf"}, top_k=8)

    # Stream the answer
    for token in rag.stream("Summarise the security section"):
        print(token, end="", flush=True)

    # Retrieval only, without calling the LLM
    passages = rag.retrieve("refund timelines")
```

You can also build the configuration in code instead of YAML:

```python
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "my_docs"},
    "llm": {"provider": "ollama", "model": "llama3.2"},
})
rag = RAGPipeline(config)
```

If the documents contain nothing relevant, the answer is `generation.abstain_message` and
`result.abstained` is `True`; the LLM is not asked to answer from nothing.

### Step 8: Improve retrieval quality

Turn these on in `rag.yaml` one at a time, and measure each change with step 9.

```yaml
retrieval:
  top_k: 5          # passages given to the LLM
  fetch_k: 30       # candidates gathered before fusion and reranking

  # Keyword + vector search. Helps with exact names, codes and part numbers.
  hybrid:
    enabled: true

  # Re-score candidates with a stronger model (needs the huggingface or cohere extra).
  rerank:
    provider: huggingface
    model: cross-encoder/ms-marco-MiniLM-L-6-v2

  # Rephrase the question before searching. Each strategy adds an LLM call.
  query_transform:
    strategies: [multi_query]    # any of: rewrite, multi_query, hyde
    num_queries: 3

  # Grade passages with the LLM and retry with a new query when none are relevant.
  corrective:
    enabled: true
    max_retries: 1
```

A sensible order is hybrid, then reranking, then query transforms, then the corrective loop. The
first two add little latency; the last two add LLM calls to every query.

### Step 9: Evaluate

Write a test set as JSON Lines, one question per line. Label which documents should be found, and
optionally give a reference answer:

```json
{"question": "How long do refunds take?", "reference_answer": "Five business days.", "relevant_sources": ["refunds.md"]}
{"question": "Who approves travel?", "reference_contexts": ["Travel must be approved by a line manager"]}
```

- `relevant_sources`: file names, paths or URLs that should appear in the results.
- `reference_contexts`: passages a correct result should contain.
- `relevant_ids`: exact `chunk_id` values, if you have them.

Run it:

```bash
ragfoundry eval eval.jsonl --output report.json                 # retrieval and generation
ragfoundry eval eval.jsonl --mode retrieval                     # fast, no LLM judging
ragfoundry eval eval.jsonl --backend ragas                      # RAGAS generation metrics
```

```
answer_relevance         0.917  (n=40)
faithfulness             0.884  (n=38)
hit_rate@5               0.950  (n=40)
mrr                      0.812  (n=40)
recall@5                 0.900  (n=40)
...
```

`n` is the number of samples actually scored. A metric is skipped for a sample when it doesn't
apply, for example faithfulness when nothing was retrieved. Per-question results, including any
metric errors, are in `report.json`.

The judge defaults to your pipeline LLM. A different model gives fairer scores, because it isn't
grading its own answers:

```yaml
evaluation:
  llm:
    provider: openai
    model: gpt-4o-mini
  k_values: [1, 3, 5]
  max_concurrency: 4
```

From Python:

```python
report = rag.evaluate("eval.jsonl", mode="all")
print(report.summary["recall@5"])     # {"mean": 0.9, "count": 40}
report.save("report.json")
```

### Step 10: Turn on tracing

Choose LangSmith or Langfuse and supply its keys:

```yaml
# LangSmith
observability:
  provider: langsmith
  api_key: ${LANGSMITH_API_KEY}
  project: my-rag
```

```yaml
# Langfuse
observability:
  provider: langfuse
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
  endpoint: https://cloud.langfuse.com    # or your self-hosted URL
```

Each ingest, query and evaluation then appears as one trace (`ragfoundry.query` and so on), with
every step and LLM call nested inside. Traces are sent in the background. Use the pipeline as a
`with` block, or call `rag.close()`, so the last traces are flushed before your program exits.

### Step 11: Use it in an application

Create the pipeline once at startup and reuse it; it holds model clients and connections.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from ragfoundry import RAGPipeline

rag = RAGPipeline.from_yaml("rag.yaml")


@asynccontextmanager
async def lifespan(app):
    yield
    rag.close()


app = FastAPI(lifespan=lifespan)


@app.get("/ask")
async def ask(q: str):
    result = await rag.aquery(q)
    return {"answer": result.answer, "sources": result.sources}
```

Every process serving the same collection must use the same `persist_dir`. It holds the chunk
record that hybrid search and incremental ingestion rely on.

### Step 12: Keep the index up to date

Re-run the same ingest whenever documents change:

```bash
ragfoundry ingest ./docs
# Documents: 42  Chunks: 320  Added: 5  Unchanged: 312  Removed: 3
```

Only new or edited chunks are embedded, and chunks that disappeared from a re-read file are
removed. If a run is interrupted, run it again and it resumes. To remove a whole source:

```bash
ragfoundry delete-source ./docs/old-policy.pdf
```

To change the embedding model, also change `collection_name` and ingest again. ragfoundry refuses
to mix two embedding models in one collection, because that silently breaks search.

---

## Configuration reference

All settings with their defaults. Unknown keys are rejected, so a typo such as `temprature` fails
immediately instead of being ignored.

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
    user_agent: ragfoundry/0.1

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

embeddings:
  provider: huggingface        # huggingface | openai | gemini
  model: sentence-transformers/all-MiniLM-L6-v2
  api_key: null                # default: the provider's environment variable
  base_url: null               # OpenAI-compatible endpoints
  options: {}                  # passed to the LangChain embeddings class

vector_store:                  # one provider block
  provider: chroma
  collection_name: ragfoundry
  # chroma:   persist_directory, or host and port
  # faiss:    path
  # pinecone: index_name (required), api_key; collection_name is used as the namespace
  # qdrant:   url and api_key, or path for local mode
  # weaviate: url, grpc_port, api_key (cloud)
  # milvus:   uri and token; omit uri for Milvus Lite
  # pgvector: connection (required)

llm:
  provider: ollama             # groq | openai | gemini | claude | ollama
  model: llama3.2
  api_key: null
  base_url: null
  temperature: null            # sent only when set; some models reject it
  max_tokens: null
  options: {}                  # passed to the LangChain chat model class

retrieval:
  top_k: 5
  fetch_k: 20
  hybrid: {enabled: false, rrf_k: 60, vector_weight: 1.0, bm25_weight: 1.0}
  rerank: null                 # {provider: huggingface | cohere, model, api_key}
  query_transform: {strategies: [], num_queries: 3}
  corrective: {enabled: false, max_retries: 1, min_relevant_docs: 1}

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

### Chunking strategies

- **fixed**: exact windows of `chunk_size` overlapping by `chunk_overlap`. Ignores document
  structure.
- **recursive**: splits on paragraphs, then lines, then words, keeping pieces under `chunk_size`.
  A good default.
- **semantic**: embeds each sentence with its neighbours and breaks where similarity drops past
  `semantic.breakpoint_percentile`. Calls the embedding model during ingestion.
- **llm**: the LLM chooses topic boundaries between numbered sentences, one call per
  `llm.window_chars` of text. Highest quality, highest cost. A reply that can't be parsed falls
  back to recursive splitting for that window, with a warning.

`chunk_size` is an upper bound for every strategy. Each chunk records `source`, `file_type`,
`start_index` and a stable `chunk_id`, plus loader metadata such as `page` (PDF), `row` (CSV) or
`sheet` and `row` (Excel).

### Extending ragfoundry

Any LangChain object can replace the configured one:

```python
rag = RAGPipeline(config, llm=my_chat_model, embeddings=my_embeddings,
                  vector_store=my_vector_store, reranker=my_reranker)
```

A reranker is any object with `rerank(query, documents, top_n) -> list[Document]`.

Add a file type or chunking strategy:

```python
from ragfoundry import register_chunker, register_loader

@register_loader(".pptx")
def pptx_loader(path, loading_config):
    return MyPptxLoader(path)          # any LangChain BaseLoader

@register_chunker("by_heading")
def heading_chunker(chunking_config, resources):
    return MyHeadingSplitter(chunk_size=chunking_config.chunk_size)   # any TextSplitter
```

Then use `chunking.strategy: by_heading` in the config.

## How a query runs

The query graph contains only the stages you enable:

```
prepare → [transform] → retrieve → [rerank] → [grade ⇄ rewrite] → generate | abstain
```

- **transform**: `rewrite` replaces the question with a clearer search query, `multi_query` adds
  alternative phrasings, and `hyde` adds a hypothetical answer passage used for vector search only.
- **retrieve**: every query runs against the vector store, and against BM25 when `hybrid` is on.
  The rankings are merged with reciprocal rank fusion.
- **rerank**: re-scores the candidates against the original question and keeps `top_k`.
- **grade / rewrite** (`corrective`): the LLM judges each passage. If fewer than
  `min_relevant_docs` pass, it writes a new query and retrieval repeats, up to `max_retries` times.
- **abstain**: when nothing relevant is found, the abstain message is returned without calling the
  LLM.

The generation prompt asks the model to cite passages as `[n]` and to treat retrieved text as data
rather than instructions.

## Evaluation details

Retrieval labels of all three kinds map onto the same metrics. When several chunks come from one
relevant document, each counts towards precision, but the document counts only once towards
recall, MAP and nDCG.

Generation metrics are judged by an LLM:

| Metric | Measures | Needs |
|---|---|---|
| `faithfulness` | Share of the answer's claims supported by the retrieved passages | retrieved passages |
| `answer_relevance` | How directly the answer addresses the question | |
| `context_relevance` | Share of retrieved passages useful for the question | retrieved passages |
| `answer_correctness` | Agreement with the reference answer | `reference_answer` |
| `context_recall` | Share of the reference answer's claims found in the passages | `reference_answer` |

The functions can be used on their own:

```python
from ragfoundry.evaluation import faithfulness, recall_at_k, ndcg_at_k

recall_at_k(["a", "x", "b"], {"a", "b"}, k=3)    # 1.0
```

## Known limitations

- **macOS with FAISS and Hugging Face models**: the `faiss-cpu` and `torch` wheels each bundle an
  OpenMP runtime, and using both in one process can abort it (`OMP: Error #15`). ragfoundry warns
  when it sees this combination. Use Chroma or Qdrant with Hugging Face models on macOS.
- **Legacy `.doc` files** are not supported; convert them to `.docx`.
- **Scanned PDFs** need OCR first. A PDF with no extractable text is reported as a failure.
- **BM25** is held in memory and scores every chunk on each query, which suits collections up to a
  few hundred thousand chunks.
- **Web crawling** stays within one site by default and respects `robots.txt`. Pages that vanish
  from a site stay indexed until you remove them with `delete-source`.
- **Pinecone** indexes must exist before ingesting; the other stores create their collection on
  first use.
- **RAGAS 0.4** imports a module that current `langchain-community` has removed. ragfoundry
  provides a placeholder for that module, and only when it is genuinely absent.

## Troubleshooting

| Message | Fix |
|---|---|
| `'X' is required for this feature. Install it with: pip install 'ragfoundry[...]'` | Run the command shown. |
| `Environment variable 'X' is referenced but not set` | Export the variable, or give a default: `${X:-value}`. |
| `This collection was indexed with embeddings '...'` | You changed the embedding model. Use a new `collection_name`, or delete `persist_dir` and the old collection, then re-ingest. |
| `No extractable text in file.pdf` | The PDF is scanned. Run OCR on it first. |
| `robots.txt disallows fetching ...` | The site forbids crawlers. Set `loading.web.respect_robots: false` only if you have permission. |
| `OMP: Error #15` on macOS | See the FAISS limitation above. |
| Answers are always the abstain message | Check `ragfoundry sources` to confirm documents are indexed; if using `corrective`, the grader may be too strict for a small model. |

## Development

```bash
git clone <your repository> && cd ragfoundry
uv venv && uv pip install -e ".[all,dev]"
pytest                    # unit tests, no network needed
pytest -m integration     # real services; see tests/test_integration.py for setup
ruff check src tests && mypy src
```

### Releasing a new version

1. Raise `version` in `pyproject.toml` and `__version__` in `src/ragfoundry/__init__.py`. PyPI
   never accepts the same version twice.
2. Build and check:

   ```bash
   rm -rf dist && uv build
   uvx twine check dist/*
   ```

3. Upload with a PyPI API token, passed through the environment so it isn't saved in your shell
   history or files:

   ```bash
   read -rs UV_PUBLISH_TOKEN && export UV_PUBLISH_TOKEN   # paste the pypi-... token
   uv publish
   unset UV_PUBLISH_TOKEN
   ```

4. Verify in a clean environment: `pip install "ragfoundry==<version>"`.
