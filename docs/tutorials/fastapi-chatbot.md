# Chatbot API with FastAPI

Serve your RAG pipeline as an HTTP API that any front end (a website, a mobile app, Slack) can
call, with a separate conversation for each user.

**Time:** 20 minutes · **Needs:** a [Groq API key](../get-started/api-keys.md).

## 1. Install

```bash
pip install "ragfoundry[loaders,huggingface,chroma,groq]" fastapi uvicorn
```

## 2. The pipeline config

```yaml title="rag.yaml"
embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2
vector_store:
  provider: chroma
  collection_name: support
llm:
  provider: groq
  model: openai/gpt-oss-120b
  temperature: 0
retrieval:
  top_k: 4
  hybrid: {enabled: true}
context_window:
  max_context_tokens: 3000
  max_history_turns: 5
```

## 3. The API

```python title="api.py"
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from ragfoundry import ChatSession, RAGPipeline

rag = RAGPipeline.from_yaml("rag.yaml")      # one pipeline for the whole server
sessions: dict[str, ChatSession] = {}       # one conversation per session id


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.ingest("docs")                       # index on startup; unchanged files are skipped
    yield
    rag.close()


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(body: Question) -> dict:
    session = sessions.setdefault(body.session_id, rag.chat())
    result = await session.aask(body.question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "abstained": result.abstained,
    }


@app.delete("/chat/{session_id}")
async def end_chat(session_id: str) -> dict:
    sessions.pop(session_id, None)
    return {"ok": True}
```

## 4. Run and call it

```bash
uvicorn api:app --reload
```

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"session_id": "alice", "question": "How long do refunds take?"}'

curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"session_id": "alice", "question": "Who approves large ones?"}'
```

FastAPI also gives you interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## 5. Streaming responses

For a typing effect in the browser, stream the answer:

```python
from fastapi.responses import StreamingResponse


@app.post("/chat/stream")
async def chat_stream(body: Question):
    session = sessions.setdefault(body.session_id, rag.chat())
    return StreamingResponse(session.astream(body.question), media_type="text/plain")
```

## Going further

**Sessions across restarts or servers.** The `sessions` dict lives in one process's memory. To
share conversations between several workers, store the history in a database or Redis and rebuild
the session for each request:

```python
saved = load_history(body.session_id)              # your storage: a list of {"role", "content"}
session = rag.chat(saved)
result = await session.aask(body.question)
save_history(body.session_id, session.to_dicts())
```

**Shared index.** Every worker must point at the same vector store and the same `persist_dir`.
For more than one machine, use a server store such as [Qdrant](../integrations/vector-stores/qdrant.md)
or [PGVector](../integrations/vector-stores/pgvector.md) and a shared volume for `persist_dir`.

See [Go to production](../how-to/production.md) for more.
