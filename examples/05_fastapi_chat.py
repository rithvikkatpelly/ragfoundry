"""A chat API with one conversation per session id.

    pip install "ragfoundry[huggingface,chroma,groq]" fastapi uvicorn
    export GROQ_API_KEY="gsk_..."
    uvicorn examples.05_fastapi_chat:app   # run from the repository root

    curl -X POST localhost:8000/chat -H 'content-type: application/json' \
         -d '{"session_id": "alice", "question": "How efficient are solar panels?"}'
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from ragfoundry import ChatSession, RAGPipeline

HERE = Path(__file__).parent
rag = RAGPipeline.from_yaml(HERE / "rag.yaml")  # one pipeline for the whole process
sessions: dict[str, ChatSession] = {}  # use Redis or a database to share across processes


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.ingest(HERE / "sample_docs")
    yield
    rag.close()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    session = sessions.setdefault(request.session_id, rag.chat())
    result = await session.aask(request.question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "standalone_question": result.standalone_question,
    }


@app.delete("/chat/{session_id}")
async def end_chat(session_id: str) -> dict:
    sessions.pop(session_id, None)
    return {"ok": True}
