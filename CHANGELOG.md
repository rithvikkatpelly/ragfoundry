# Changelog

## 0.2.0

### Added

- **Chat with conversation history.** `rag.chat()` returns a `ChatSession` with `ask`, `aask`,
  `stream`, `astream`, `reset` and `to_dicts`. `query`, `aquery`, `stream`, `astream` and
  `retrieve` accept `history=` as LangChain messages, `{"role", "content"}` dicts or
  `(role, content)` tuples.
- **Follow-up question rewriting.** With chat history, follow-ups are rewritten into standalone
  questions before retrieval. The rewritten question is in `RAGResult.standalone_question`.
- **Context window limits** in a new `context_window` config section:
  - `max_context_tokens` caps the retrieved passages sent to the LLM, shortening the passage that
    crosses the budget (`truncate_last_passage`).
  - `max_history_turns` and `max_history_tokens` limit how much chat history is sent.
  - `token_counter` is `approximate` or `tiktoken`, or pass any function as
    `RAGPipeline(token_counter=...)`.
- `RAGResult.context_tokens` reports the tokens of context sent to the LLM.
- `pack_documents` and `window_history` are exported for use in your own code.
- MIT license, project links on PyPI, six example scripts and sample documents.

### Changed

- `delete_source` accepts relative paths and `Path` objects, matching how `ingest` resolves them.
- The LLM is created only when a stage first needs it, so retrieval works with no `llm` configured.
- The README is rewritten as a full guide to every feature.

### Removed

- The `ragfoundry` command-line interface and the `typer` dependency. Use the Python API, with
  `RAGPipeline.from_yaml("rag.yaml")` for YAML configuration.

## 0.1.0

First release.
