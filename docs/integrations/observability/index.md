# Observability

Send a trace of every ingest, query and evaluation, with each step, prompt and LLM reply, to a
tracing service. How to use traces: [Trace with LangSmith or Langfuse](../../how-to/tracing.md).

| Provider | Hosting | Extra |
|---|---|---|
| [LangSmith](langsmith.md) | Cloud (by LangChain) | `langsmith` |
| [Langfuse](langfuse.md) | Cloud or self-hosted, open source | `langfuse` |

| Setting | Default | Meaning |
|---|---|---|
| `provider` | `none` | `none`, `langsmith` or `langfuse` |
| `project` | `ragfoundry` | LangSmith project name |
| `api_key` | | LangSmith API key |
| `public_key`, `secret_key` | | Langfuse keys |
| `endpoint` | | Self-hosted or regional server URL |
| `tags` | `[]` | Labels added to every trace |
