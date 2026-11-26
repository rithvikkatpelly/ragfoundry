# Troubleshooting

## Error messages

| Message | Cause and fix |
|---|---|
| `'X' is required for this feature. Install it with: pip install 'ragfoundry[...]'` | A provider's extra isn't installed. Run the command shown. |
| `No LLM configured: set 'llm' or pass llm=...` | Answering needs an `llm` section. Retrieval alone doesn't. |
| `Environment variable 'X' is referenced but not set` | Your YAML uses `${X}`. Export it, or give a default: `${X:-value}`. |
| `Extra inputs are not permitted` | A config key is misspelled or in the wrong section. Check the [configuration reference](configuration.md). |
| `chunk_overlap must be smaller than chunk_size` | Lower `chunk_overlap`. |
| `fetch_k must be greater than or equal to top_k` | Raise `fetch_k`. |
| `This collection was indexed with embeddings '...'` | You changed the embedding model. Use a new `collection_name`, or delete `persist_dir` and the old collection, then ingest again. |
| `No extractable text in file.pdf` | A scanned PDF. Run OCR first (`ocrmypdf in.pdf out.pdf`). |
| `Unsupported file type '.doc'` | Convert to a supported format, or [register a loader](../how-to/custom-components.md#add-a-file-type). |
| `No supported files found in directory ...` | The folder (or your `loading.glob`) has no supported files. |
| `Source not found: ...` | Check the path; relative paths are relative to where you run Python. |
| `robots.txt disallows fetching ...` | The site forbids crawlers. Only set `respect_robots: false` with permission. |
| `Unknown chunking strategy '...'` | Use `fixed`, `recursive`, `semantic`, `llm`, or register your own. |
| Model not found / decommissioned | Providers retire models; choose a current one from their model list. |
| `401` / authentication errors | Wrong or missing API key. Check the environment variable name on the provider's page. |
| `429` / rate limit errors | Too many requests. Slow down, lower `evaluation.max_concurrency`, or upgrade your plan. |
| `OMP: Error #15` on macOS | FAISS with Hugging Face models in one process. Use Chroma or Qdrant. |

## Answers are wrong or unhelpful

Work through these in order:

1. **Is the document indexed?** `print(rag.sources())`.
2. **Was the right passage retrieved?** Print `result.documents`. If the passage isn't there, it's a
   retrieval problem:
    - turn on [hybrid search](../how-to/hybrid-search.md) for names, codes and exact terms,
    - add a [reranker](../how-to/reranking.md),
    - try a smaller [chunk size](../how-to/chunk-size.md), or a stronger
      [embedding model](../integrations/embeddings/index.md),
    - raise `top_k`.
3. **The passage was retrieved but the answer is still wrong?** It's a generation problem: try a
   stronger LLM, a clearer [system prompt](../how-to/prompts.md), or check whether a
   [context budget](../how-to/context-window.md) truncated the passage
   (`metadata["truncated"]`).
4. **Measure** with an [evaluation](../how-to/evaluation.md) so you know a fix helped.

## Always "I don't have enough information"

- Check documents are indexed: `rag.sources()`.
- With [corrective retrieval](../how-to/corrective-retrieval.md) on, the grader may be rejecting
  good passages. Try turning it off or using a stronger model.
- A [metadata filter](../how-to/metadata-filters.md) may match nothing. `source` values are
  absolute paths; copy them from `rag.sources()`.

## Follow-up questions go wrong

Print `result.standalone_question`. If the rewrite is poor, use a stronger LLM; if you don't need
follow-ups, set `context_window.condense_question: false`.

## Slow

- First run: models are downloading. Later runs use the cache.
- Ingesting on a CPU with Hugging Face embeddings: use a GPU, a smaller model, or hosted
  embeddings.
- Each query: query transforms and corrective retrieval add LLM calls; a reranker adds model
  passes; long prompts slow answers (set a [context budget](../how-to/context-window.md)).

## Still stuck?

[Open an issue](https://github.com/rithvikkatpelly/ragfoundry/issues) with your configuration
(without keys), the full error, and your ragfoundry version
(`python -c "import ragfoundry; print(ragfoundry.__version__)"`).
