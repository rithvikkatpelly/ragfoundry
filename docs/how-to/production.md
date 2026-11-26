# Go to production

A checklist for running ragfoundry behind a real application.

## Architecture

- **Create one pipeline per process** at startup and share it. It holds model clients,
  connections and the loaded embedding model.
- **Create one `ChatSession` per conversation.** Persist `session.to_dicts()` in your database if
  conversations must survive restarts or move between servers.
- **Separate ingestion from serving.** Run `ingest` from a scheduled job or worker, not inside web
  requests.
- **Use a server vector store** ([Qdrant](../integrations/vector-stores/qdrant.md),
  [PGVector](../integrations/vector-stores/pgvector.md), [Pinecone](../integrations/vector-stores/pinecone.md),
  [Weaviate](../integrations/vector-stores/weaviate.md), [Milvus](../integrations/vector-stores/milvus.md))
  when more than one process reads the index.
- **Share `persist_dir`** between the ingesting job and every serving process (a shared volume).
  It holds the chunk record used for incremental ingestion and hybrid search.

## Configuration

- Keep config in YAML, one file per environment, with secrets from environment variables:
  `api_key: ${OPENAI_API_KEY}`.
- Pin versions: `ragfoundry==0.2.0` and your provider packages, so upgrades are deliberate.
- Set `llm.temperature: 0` and `llm.max_tokens` for predictable, bounded answers.
- Set `context_window.max_context_tokens` and `max_history_tokens` to bound cost per request.

## Quality

- Keep an [evaluation](evaluation.md) set of real user questions and run it in CI before changing
  models, prompts or retrieval settings.
- Turn on [tracing](tracing.md) to inspect bad answers.
- Log unanswered questions. With [corrective retrieval](corrective-retrieval.md) on,
  `result.abstained` marks them; a rising rate means questions your documents don't cover.

## Performance

| Lever | Effect |
|---|---|
| Hosted embeddings (OpenAI, Gemini) | No local model to load; scales with no GPU |
| Smaller `top_k`, a `max_context_tokens` budget | Shorter prompts: faster and cheaper answers |
| Reranking | Better passages, plus 100 to 500 ms |
| Query transforms, corrective retrieval | Better on hard questions, but extra LLM calls each |
| `aquery` / `astream` | Many concurrent requests per process |

## Security

- The default prompt tells the LLM to treat document text as data. If you write your own
  [system prompt](prompts.md), keep that rule: documents can contain text designed to manipulate
  the model.
- Ingest only documents that every user of the index is allowed to see, or use
  [metadata filters](metadata-filters.md) per user or team, applied in your server code (never
  from user input directly).
- Keep API keys in a secret manager, never in config files committed to Git.
- Web crawling respects `robots.txt` by default; keep it on.

## Deploying

ragfoundry is a normal Python library, so it deploys like any Python service: a container, a VM,
a serverless platform with enough memory for your embedding model, or Streamlit Community Cloud for
a [Streamlit app](../tutorials/streamlit-app.md). With local Hugging Face embeddings, allow about
1 GB of memory and pre-download the model in your image to avoid a slow first request.
