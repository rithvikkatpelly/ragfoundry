# API keys

The LLM (and, if you choose a hosted one, the embedding model) runs on a provider's servers. They
identify you with an **API key**: a long secret string, like a password.

## Get a free key from Groq

Groq offers a free tier and fast open models, which makes it a good first choice.

1. Go to [console.groq.com/keys](https://console.groq.com/keys) and sign in.
2. Click **Create API Key**, give it a name, and copy the key (it starts with `gsk_`).

## Give the key to ragfoundry

Each provider's library reads its key from an **environment variable**, so the key never has to
appear in your code.

=== "macOS / Linux"

    ```bash
    export GROQ_API_KEY="gsk_..."
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:GROQ_API_KEY = "gsk_..."
    ```

=== ".env file"

    Create a file called `.env` in your project folder:

    ```bash
    GROQ_API_KEY=gsk_...
    ```

    Load it at the top of your script (`pip install python-dotenv`):

    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```

=== "Google Colab"

    Click the key icon in the left sidebar, add a secret named `GROQ_API_KEY`, and switch on
    *Notebook access*. Then pass it in your config:

    ```python
    from google.colab import userdata
    llm = {"provider": "groq", "model": "openai/gpt-oss-120b",
           "api_key": userdata.get("GROQ_API_KEY")}
    ```

An `export` only lasts for the current terminal. A `.env` file lasts until you delete it.

!!! danger "Keep keys secret"
    Never commit keys to Git or paste them into chats, issues or screenshots. Add `.env` to your
    `.gitignore`. If a key leaks, delete it in the provider's console and create a new one.

## Keys for every provider

| Provider | Environment variable | Get a key | Free tier |
|---|---|---|---|
| [Groq](../integrations/llms/groq.md) | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) | Yes |
| [Gemini](../integrations/llms/gemini.md) | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) | Yes |
| [OpenAI](../integrations/llms/openai.md) | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | No |
| [Claude](../integrations/llms/claude.md) | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | No |
| [Pinecone](../integrations/vector-stores/pinecone.md) | `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io/) | Yes |
| [Cohere Rerank](../integrations/rerankers/cohere.md) | `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com/) | Trial |

## No API key: run models locally

[Hugging Face](../integrations/embeddings/huggingface.md) embedding models run on your computer
for free and need no key. For the LLM, [Ollama](../integrations/llms/ollama.md) runs open models
locally:

1. Install Ollama from [ollama.com](https://ollama.com).
2. Download a model: `ollama pull llama3.2`
3. Use `{"provider": "ollama", "model": "llama3.2"}` as your LLM.

Local models are private and free, but slower than hosted ones unless you have a good GPU. See the
[fully local tutorial](../tutorials/local-rag.md).

## Next

[Build your first RAG pipeline](first-pipeline.md).
