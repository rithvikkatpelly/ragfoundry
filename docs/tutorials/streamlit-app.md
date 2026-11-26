# Streamlit chat app

Build a web app where users upload documents and chat with them. Answers show their sources, and
follow-up questions work. The whole app is about 110 lines of Python plus a config file.

**Time:** 20 minutes · **Needs:** a [Groq API key](../get-started/api-keys.md).

<figure markdown>
```text
┌──────────────────────┬─────────────────────────────────────────────────┐
│ Documents            │ 📄 Chat with your documents                     │
│ ┌──────────────────┐ │                                                 │
│ │ Upload files     │ │  🧑 What does a sourdough starter need?          │
│ │ Or a web page URL│ │  🤖 Regular feeding with flour and water [1].    │
│ │ [Add to KB]      │ │     ▸ Sources (2)                               │
│ └──────────────────┘ │  🧑 How often?                                   │
│ Indexed (2)          │  🤖 Usually once a day at room temperature [1].  │
│  solar.md · 1   ✕    │                                                 │
│  sourdough.md · 1 ✕  │ ┌─────────────────────────────────────────────┐ │
│ [New conversation]   │ │ Ask a question about your documents         │ │
└──────────────────────┴─┴─────────────────────────────────────────────┴─┘
```
</figure>

## 1. Set up the project

```bash
mkdir rag-chat-app && cd rag-chat-app
python -m venv .venv && source .venv/bin/activate
```

```text title="requirements.txt"
ragfoundry[pdf,docx,excel,web,huggingface,chroma,groq,ollama]>=0.2
streamlit>=1.40
python-dotenv>=1.0
```

```bash
pip install -r requirements.txt
```

```bash title=".env"
GROQ_API_KEY=gsk_...
```

Add `.env`, `storage/` and `uploads/` to `.gitignore` so your key and data are never committed.

## 2. Configure the pipeline in YAML

Keeping the pipeline in a file means you can switch models or vector stores without touching the
app code.

```yaml title="rag.yaml"
persist_dir: storage

embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  provider: chroma
  collection_name: documents

llm:
  provider: groq
  model: ${GROQ_MODEL:-openai/gpt-oss-120b}
  temperature: 0
  # Fully local instead: provider: ollama, model: llama3.2

chunking:
  strategy: recursive
  chunk_size: 800
  chunk_overlap: 150

loading:
  on_error: skip            # a bad file is reported instead of stopping the upload

retrieval:
  top_k: 4
  fetch_k: 20
  hybrid:
    enabled: true           # keyword + vector search

context_window:
  max_context_tokens: 3000  # retrieved text sent per question
  max_history_turns: 5      # earlier chat turns sent per question
```

- `persist_dir: storage` keeps the index in a `storage/` folder inside the project.
- `${GROQ_MODEL:-openai/gpt-oss-120b}` reads the model from the `GROQ_MODEL` environment
  variable, or uses `openai/gpt-oss-120b` if it isn't set.
- `on_error: skip` means one unreadable upload is reported instead of failing the whole batch.

## 3. Write the app

```python title="app.py"
"""Chat with your documents: a Streamlit front end for a ragfoundry pipeline."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from ragfoundry import RAGPipeline

load_dotenv()
UPLOADS = Path("uploads")
SUPPORTED = ["pdf", "docx", "txt", "md", "csv", "xlsx", "xls"]

st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_yaml(os.environ.get("RAG_CONFIG", "rag.yaml"))


rag = get_pipeline()
if "chat" not in st.session_state:
    st.session_state.chat = rag.chat()
    st.session_state.messages = []  # [{"role", "content", "sources"}] for display

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Documents")

    if rag.config.llm.provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        key = st.text_input("Groq API key", type="password", help="console.groq.com/keys")
        if not key:
            st.info("Enter your Groq API key to start.")
            st.stop()
        os.environ["GROQ_API_KEY"] = key

    with st.form("add", clear_on_submit=True):
        files = st.file_uploader("Upload files", type=SUPPORTED, accept_multiple_files=True)
        url = st.text_input("Or a web page URL")
        if st.form_submit_button("Add to knowledge base", type="primary") and (files or url):
            UPLOADS.mkdir(exist_ok=True)
            sources = [str(UPLOADS / f.name) for f in files or []]
            for file, path in zip(files or [], sources):
                Path(path).write_bytes(file.getbuffer())
            sources += [url] if url else []
            with st.spinner("Reading and indexing..."):
                report = rag.ingest(sources)
            st.success(f"Added {report.chunks_added} chunks ({report.chunks_unchanged} unchanged)")
            for failure in report.failures:
                st.error(f"{Path(failure.source).name}: {failure.error}")

    indexed = rag.sources()
    st.subheader(f"Indexed ({len(indexed)})")
    for source, chunks in indexed.items():
        name, remove = st.columns([5, 1])
        name.caption(f"{Path(source).name if '://' not in source else source} · {chunks} chunks")
        if remove.button("✕", key=source, help="Remove from the knowledge base"):
            rag.delete_source(source)
            if Path(source).parent == UPLOADS.resolve():
                Path(source).unlink(missing_ok=True)
            st.rerun()

    if st.button("New conversation", use_container_width=True):
        st.session_state.chat.reset()
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------- chat
st.title("📄 Chat with your documents")


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for number, doc in enumerate(sources, start=1):
                page = f", page {doc['page'] + 1}" if isinstance(doc.get("page"), int) else ""
                st.markdown(f"**[{number}] {doc['name']}{page}**")
                st.caption(doc["text"][:400])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

if not indexed:
    st.info("Add documents in the sidebar, then ask a question about them.")
elif question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chat.ask(question)
        st.markdown(result.answer)
        sources = [
            {
                "name": Path(doc.metadata.get("source", "")).name or doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "text": doc.page_content,
            }
            for doc in result.documents
        ]
        show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": sources}
    )
```

## 4. How it works

**One pipeline for all users.** `@st.cache_resource` builds the pipeline once per server process.
Streamlit re-runs the whole script on every click; without the cache it would reload the
embedding model each time.

**One conversation per browser tab.** `st.session_state` is separate for each visitor, so each
gets their own `rag.chat()` session. `st.session_state.messages` keeps what's shown on screen,
including sources; the `ChatSession` keeps what's sent to the LLM.

**Asking for the key.** If `GROQ_API_KEY` isn't set, the sidebar shows a password field and
`st.stop()` halts the page until a key is entered. The key is then put in the environment, where
Groq's client reads it the first time the LLM is used.

**Uploading.** Streamlit gives uploaded files as in-memory objects. They are written to `uploads/`
and passed to `rag.ingest()` together with any URL. The form's `clear_on_submit` empties the
uploader afterwards, so files aren't indexed twice. Re-uploading an unchanged file costs nothing,
because ingestion is incremental.

**Sources.** `rag.sources()` lists indexed documents with their chunk counts. ✕ calls
`rag.delete_source()` and removes the uploaded file.

**Answers.** `chat.ask()` returns a `RAGResult`; its `documents` are the passages the LLM read,
shown in the "Sources" expander with page numbers for PDFs.

## 5. Run it

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload some files, click **Add to knowledge
base**, and ask a question. The first start downloads the embedding model, which takes a minute.

## Ideas to extend it

- **Stream answers:** replace the spinner and `chat.ask` with
  `answer = st.write_stream(st.session_state.chat.stream(question))`. Streaming doesn't return
  sources, so drop the expander or call `rag.retrieve()` for them.
- **Better retrieval:** add a [reranker](../how-to/reranking.md) to `rag.yaml`.
- **Run without keys:** switch `llm` to [Ollama](../integrations/llms/ollama.md).
- **Deploy:** Streamlit Community Cloud, a container, or any VM. Use a server-based
  [vector store](../integrations/vector-stores/index.md) such as Qdrant if several app instances
  share one index.

## Next

[Chatbot API with FastAPI](fastapi-chatbot.md) · [Chat with history](../how-to/chat-history.md)
