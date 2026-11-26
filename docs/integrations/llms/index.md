# Chat models (LLMs)

The LLM writes answers from the retrieved passages. It's also used, when you turn them on, for
chat follow-up rewriting, query transforms, corrective grading, LLM chunking and evaluation.

```python
"llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0}
```

| Provider | Runs | Key | Extra | Free tier |
|---|---|---|---|---|
| [Groq](groq.md) | Hosted | `GROQ_API_KEY` | `groq` | Yes |
| [OpenAI](openai.md) | Hosted | `OPENAI_API_KEY` | `openai` | No |
| [Gemini](gemini.md) | Hosted | `GOOGLE_API_KEY` | `gemini` | Yes |
| [Claude](claude.md) | Hosted | `ANTHROPIC_API_KEY` | `claude` | No |
| [Ollama](ollama.md) | Your machine | None | `ollama` | Free |

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `provider` | | One of the above |
| `model` | | The provider's model name |
| `temperature` | not sent | 0 for the most consistent answers. Some reasoning models reject it, so it's only sent when set. |
| `max_tokens` | provider default | Maximum answer length |
| `api_key` | the provider's environment variable | API key |
| `base_url` | | Custom endpoint (OpenAI-compatible servers, a proxy, remote Ollama) |
| `options` | `{}` | Extra arguments for the LangChain chat model class |

The LLM is created the first time it's needed, so scripts that only ingest or retrieve don't need
an `llm` section or key.

!!! tip "Model names change"
    Providers add and retire models often. If you get "model not found" or "decommissioned", pick
    a current one from the provider's model list, linked on each page.

## A different model for evaluation

The LLM judging [evaluations](../../how-to/evaluation.md) can be different from the one answering:

```python
"evaluation": {"llm": {"provider": "openai", "model": "gpt-4o-mini"}}
```

Other providers (Azure OpenAI, AWS Bedrock, Mistral, Together...) can be passed in as any
LangChain chat model; see [Custom components](../../how-to/custom-components.md).
