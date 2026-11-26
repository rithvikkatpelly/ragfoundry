# Installation

## 1. Check Python

ragfoundry needs **Python 3.10 or newer**. In a terminal:

```bash
python --version        # on some systems: python3 --version
```

If it's older than 3.10 or not installed, download the latest version from
[python.org](https://www.python.org/downloads/).

## 2. Create a project folder and virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

=== "macOS / Linux"

    ```bash
    mkdir my-rag && cd my-rag
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    mkdir my-rag; cd my-rag
    python -m venv .venv
    .venv\Scripts\activate
    ```

Your prompt now starts with `(.venv)`. Run the `activate` line again whenever you open a new
terminal for this project.

## 3. Install ragfoundry with extras

ragfoundry's core is small. Support for each provider is an optional **extra**, written in square
brackets, so you install only what you use.

**Recommended for beginners**: free local embeddings, a local vector store, and Groq's free LLM
API, reading PDFs:

```bash
pip install "ragfoundry[pdf,huggingface,chroma,groq]"
```

**Fully offline**, using Ollama on your own computer (no API keys at all):

```bash
pip install "ragfoundry[pdf,huggingface,chroma,ollama]"
```

### All extras

| Category | Extras |
|---|---|
| File types | `pdf`, `docx`, `excel`, `web`, or `loaders` for all of them. TXT, Markdown and CSV need no extra. |
| Embeddings | `huggingface`, `openai`, `gemini` |
| Vector stores | `chroma`, `faiss`, `pinecone`, `qdrant`, `weaviate`, `milvus`, `pgvector` |
| LLMs | `groq`, `openai`, `gemini`, `claude`, `ollama` |
| Reranking | `huggingface` (local) or `cohere` |
| Token counting | `tokens` |
| Evaluation | `ragas` |
| Tracing | `langsmith`, `langfuse` |
| Everything | `all` |

Combine as many as you need, separated by commas. You can add more later by running `pip install`
again with the new extras.

!!! note "Forgot an extra?"
    If you choose a provider whose extra isn't installed, ragfoundry stops with a message showing
    the exact `pip install` command to run.

## 4. Check it works

```bash
python -c "import ragfoundry; print(ragfoundry.__version__)"
```

You should see a version number such as `0.2.0`.

## Google Colab

In a Colab notebook cell:

```python
%pip install -q "ragfoundry[pdf,huggingface,chroma,groq]"
```

If Colab warns about conflicting packages, choose **Runtime → Restart session** and continue.

## Next

[Get an API key](api-keys.md), or skip to [the local option](api-keys.md#no-api-key-run-models-locally).
