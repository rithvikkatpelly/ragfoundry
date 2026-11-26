# Fully local RAG with Ollama

Build a pipeline where documents, embeddings and the LLM all stay on your computer. No API keys,
no costs, and it works offline after the first download.

**Time:** 15 minutes, plus model downloads · **Needs:** about 8 GB of RAM; a GPU or Apple Silicon
makes answers much faster.

## 1. Install Ollama and a model

1. Download Ollama from [ollama.com](https://ollama.com) and start it.
2. Pull a model:

    ```bash
    ollama pull llama3.2          # 2 GB, runs on most laptops
    # or a stronger one if you have the memory:
    ollama pull qwen3:8b          # 5 GB
    ```

3. Check it's running: `ollama list` shows the model.

## 2. Install ragfoundry

```bash
mkdir local-rag && cd local-rag
python -m venv .venv && source .venv/bin/activate
pip install "ragfoundry[loaders,huggingface,chroma,ollama]"
```

## 3. Build the pipeline

```python title="local_rag.py"
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"},
    "vector_store": {"provider": "chroma", "collection_name": "local"},
    "llm": {"provider": "ollama", "model": "llama3.2", "temperature": 0},
    "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
    "context_window": {"max_context_tokens": 2000},
})

with RAGPipeline(config) as rag:
    print(rag.ingest("docs").to_dict())
    chat = rag.chat()
    while question := input("\nYou: ").strip():
        print("Assistant: ", end="", flush=True)
        for token in chat.stream(question):
            print(token, end="", flush=True)
        print()
```

Put your files in a `docs/` folder and run `python local_rag.py`.

Choices explained:

- **`BAAI/bge-small-en-v1.5`** is a small, accurate English embedding model (384 dimensions). It
  downloads once from Hugging Face, then runs offline.
- **`max_context_tokens: 2000`** keeps prompts short. Local models slow down a lot with long
  prompts, and small models get confused by too much text.

## Reasoning models

Models such as `qwen3` "think" before answering, which is slow and can leak the thinking into the
answer. Turn it off through `options`, which are passed to LangChain's `ChatOllama`:

```python
"llm": {"provider": "ollama", "model": "qwen3:8b", "options": {"reasoning": False}}
```

## Ollama on another machine

Point `base_url` at it:

```python
"llm": {"provider": "ollama", "model": "llama3.2", "base_url": "http://192.168.1.20:11434"}
```

## Tips for small local models

- Keep `top_k` at 3 to 5. More passages make small models worse, not better.
- Avoid [corrective retrieval](../how-to/corrective-retrieval.md) and many
  [query transforms](../how-to/query-transforms.md); each adds a slow LLM call.
- If follow-up questions go wrong, check `result.standalone_question`. Small models sometimes
  rewrite poorly; set `context_window.condense_question: false` to search with the question as
  typed.

## Next

[Ollama integration reference](../integrations/llms/ollama.md) ·
[Hugging Face embeddings](../integrations/embeddings/huggingface.md)
