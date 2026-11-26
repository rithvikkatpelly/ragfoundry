# Chat with a PDF

Build a terminal chatbot that answers questions about a PDF, remembers the conversation and shows
which page each answer came from.

**Time:** 15 minutes · **Needs:** a [Groq API key](../get-started/api-keys.md) and any PDF.

## 1. Set up

```bash
mkdir pdf-chat && cd pdf-chat
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install "ragfoundry[pdf,huggingface,chroma,groq]"
export GROQ_API_KEY="gsk_..."                         # Windows: $env:GROQ_API_KEY="gsk_..."
```

Copy a PDF into the folder, for example `handbook.pdf`.

## 2. Configure the pipeline

```python title="pdf_chat.py"
import sys
from pathlib import Path

from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "vector_store": {"provider": "chroma", "collection_name": "pdf_chat"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0},
    "chunking": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 150},
    "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
    "context_window": {"max_context_tokens": 3000, "max_history_turns": 5},
})
```

Why these settings:

- `chunk_size: 800` keeps each chunk to roughly a paragraph or two, which suits most PDFs.
- `hybrid` adds keyword search, so exact terms such as product names or section numbers are found.
- `max_context_tokens: 3000` caps how much PDF text is sent per question, keeping requests fast
  and cheap.
- `max_history_turns: 5` sends the last five questions and answers, so follow-ups work.

## 3. Index the PDF

Add to `pdf_chat.py`:

```python
pdf = sys.argv[1] if len(sys.argv) > 1 else "handbook.pdf"
rag = RAGPipeline(config)
report = rag.ingest(pdf)
print(f"Indexed {report.documents_loaded} pages as {report.chunks_total} chunks")
```

The PDF loader creates one document per page and records its page number in the metadata. Running
the script again skips pages that haven't changed.

## 4. Chat loop with page citations

```python
chat = rag.chat()
print("Ask about the PDF. Press Enter on an empty line to quit.")

while question := input("\nYou: ").strip():
    result = chat.ask(question)
    print(f"\nAssistant: {result.answer}")
    for number, doc in enumerate(result.documents, start=1):
        page = doc.metadata.get("page")
        where = f"page {page + 1}" if page is not None else "unknown page"
        print(f"  [{number}] {Path(doc.metadata['source']).name}, {where}")

rag.close()
```

PDF page numbers start at 0 in the metadata, so `page + 1` shows the number a reader expects.

## 5. Run it

```bash
python pdf_chat.py handbook.pdf
```

```text
Indexed 12 pages as 31 chunks
Ask about the PDF. Press Enter on an empty line to quit.

You: How long do refunds take?

Assistant: Refunds are processed within five business days [1].
  [1] handbook.pdf, page 3
  [2] handbook.pdf, page 4

You: Who has to approve large ones?

Assistant: Refunds over 500 dollars need approval from a finance manager [1].
  [1] handbook.pdf, page 3
```

"Large ones" was understood as "large refunds" because the chat rewrote the follow-up into a
standalone question before searching. You can print it with `result.standalone_question`.

## 6. Stream the answer (optional)

To print the answer word by word instead of waiting for all of it, replace the `chat.ask` line and
the print after it with:

```python
print("\nAssistant: ", end="")
for token in chat.stream(question):
    print(token, end="", flush=True)
print()
```

Streaming doesn't return the source documents. Use `ask` when you need citations.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No extractable text in handbook.pdf` | The PDF is a scanned image. Run it through OCR first (for example `ocrmypdf in.pdf out.pdf`). |
| Answers say there is not enough information | Print `result.documents` to see what was retrieved. Try a smaller `chunk_size`, or turn on [reranking](../how-to/reranking.md). |
| Model not found | Groq changes models often; pick one from [console.groq.com/docs/models](https://console.groq.com/docs/models). |

## Next

- Put a web interface on it: [Streamlit chat app](streamlit-app.md)
- Measure answer quality: [Evaluate and improve your RAG](evaluate-and-improve.md)
