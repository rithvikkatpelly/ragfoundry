# Embedding models

An embedding model turns text into a vector of numbers so passages with similar meaning can be
found. Choose one with the `embeddings` section:

```python
"embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}
```

| Provider | Runs | Cost | Key | Extra |
|---|---|---|---|---|
| [Hugging Face](huggingface.md) | On your machine | Free | None | `huggingface` |
| [OpenAI](openai.md) | Hosted | Per token | `OPENAI_API_KEY` | `openai` |
| [Gemini](gemini.md) | Hosted | Free tier, then per token | `GOOGLE_API_KEY` | `gemini` |

## Settings

| Setting | Meaning |
|---|---|
| `provider` | `huggingface`, `openai` or `gemini` |
| `model` | The model name |
| `api_key` | Overrides the provider's environment variable |
| `base_url` | OpenAI only: an OpenAI-compatible server |
| `options` | Extra arguments for the LangChain embeddings class |

## Important: pick once per collection

Questions and chunks must be embedded by the same model. A collection records the model that
filled it, and ragfoundry refuses to use a different one with it. To switch model, use a new
`vector_store.collection_name` and ingest again.

## Popular models

| Model | Provider | Dimensions | Notes |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | 384 | Tiny and fast; good English baseline |
| `BAAI/bge-small-en-v1.5` | Hugging Face | 384 | Better English retrieval at the same size |
| `BAAI/bge-base-en-v1.5` | Hugging Face | 768 | Stronger, slower |
| `intfloat/multilingual-e5-base` | Hugging Face | 768 | Many languages |
| `text-embedding-3-small` | OpenAI | 1536 | Cheap, strong, multilingual |
| `text-embedding-3-large` | OpenAI | 3072 | Strongest OpenAI model |
| `models/gemini-embedding-001` | Gemini | 3072 | Multilingual |

Dimensions matter for [Pinecone](../vector-stores/pinecone.md), where you create the index
yourself. Other stores detect it automatically.

Using another provider (Cohere, Mistral, Bedrock, Azure)? Pass any LangChain `Embeddings` object:
see [Custom components](../../how-to/custom-components.md).
