# Your first RAG pipeline

In this guide you will build a program that reads a few documents and answers questions about
them. It takes about 10 minutes.

**Before you start:** [install ragfoundry](installation.md) with
`pip install "ragfoundry[pdf,huggingface,chroma,groq]"` and [set your Groq key](api-keys.md).

## Step 1: Add some documents

In your project folder, create a folder called `docs` with a file `docs/company.md`:

```markdown title="docs/company.md"
# Acme Corp policies

Refunds are issued within five business days of receiving the returned item.
Refunds over 500 dollars must be approved by a finance manager.

Employees get 25 days of paid holiday per year. Unused days can be carried over,
up to a maximum of 5 days.

The office is open from 8am to 6pm, Monday to Friday.
```

You can also put PDFs, Word files, text files, CSVs or spreadsheets in this folder.

## Step 2: Write the program

Create `rag.py` next to the `docs` folder:

```python title="rag.py"
from ragfoundry import RAGConfig, RAGPipeline

# 1. Describe the pipeline
config = RAGConfig.model_validate({
    "embeddings": {                                    # (1)!
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "vector_store": {                                  # (2)!
        "provider": "chroma",
        "collection_name": "company_docs",
    },
    "llm": {                                           # (3)!
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "temperature": 0,
    },
})

# 2. Build it
rag = RAGPipeline(config)

# 3. Index the documents
report = rag.ingest("docs")                            # (4)!
print(f"Indexed {report.chunks_total} chunks from {report.documents_loaded} documents")

# 4. Ask a question
result = rag.query("How long do refunds take?")        # (5)!
print("Answer:", result.answer)
print("Sources:", result.sources)

rag.close()                                            # (6)!
```

1. The **embedding model** turns text into vectors so passages with similar meaning can be found.
   This one is free and runs on your computer; it downloads (about 90 MB) the first time.
2. The **vector store** keeps the vectors. Chroma stores them in a local folder, `.ragfoundry/`,
   so there is nothing to set up. `collection_name` names this set of documents.
3. The **LLM** writes the answers. Groq reads your key from `GROQ_API_KEY`. `temperature: 0` makes
   answers as consistent as possible.
4. `ingest` reads every supported file in `docs`, splits it into chunks, embeds them and stores
   them. Running it again only processes files that changed.
5. `query` finds the most relevant chunks and asks the LLM to answer from them.
6. `close` releases database connections. You can also use `with RAGPipeline(config) as rag:`.

## Step 3: Run it

```bash
python rag.py
```

```text
Indexed 1 chunks from 1 documents
Answer: Refunds are issued within five business days of receiving the returned item [1].
Sources: ['/home/you/my-rag/docs/company.md']
```

The `[1]` is a citation: it refers to the first passage the LLM was given. `result.sources` lists
the files those passages came from.

!!! success "That's a working RAG system"
    Your documents are indexed, retrieved and used to answer. Everything else in these docs makes
    it better, bigger or easier to use.

## Step 4: Look at what was retrieved

Add this before `rag.close()` to see the passages the LLM read:

```python
for number, doc in enumerate(result.documents, start=1):
    print(f"[{number}] {doc.metadata['source']}")
    print(doc.page_content[:200])
```

If an answer is wrong, check here first. Usually the right passage wasn't retrieved; the
[how-to guides](../how-to/index.md) show how to improve retrieval.

## Step 5: Ask something that isn't in the documents

```python
result = rag.query("Who is the CEO?")
print(result.answer)
```

```text
The provided documents don't mention who the CEO is.
```

The LLM is told to answer only from the passages and to say so when the answer isn't there. This
keeps it from making things up.

Search always returns the closest passages, even when none of them is relevant, so here the LLM
saw your policy text and said it doesn't contain the answer. To skip the LLM entirely when no
passage is relevant, turn on [corrective retrieval](../how-to/corrective-retrieval.md): passages
are checked first, and if none is relevant the pipeline returns a fixed message and sets
`result.abstained` to `True`.

## Step 6: Have a conversation

`rag.chat()` remembers earlier questions, so follow-ups work:

```python
chat = rag.chat()
print(chat.ask("How many holiday days do employees get?").answer)
print(chat.ask("Can they carry them over?").answer)     # "they" = employees, "them" = holiday days
```

## Step 7: Move the settings to a file (optional)

The same configuration can live in a YAML file, which is easier to change without touching code:

```yaml title="rag.yaml"
embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2
vector_store:
  provider: chroma
  collection_name: company_docs
llm:
  provider: groq
  model: openai/gpt-oss-120b
  temperature: 0
```

```python
rag = RAGPipeline.from_yaml("rag.yaml")
```

## What you learned

- A pipeline is described by a **configuration** with three required parts: `embeddings`,
  `vector_store` and `llm`.
- `ingest()` indexes documents; `query()` answers; `chat()` holds a conversation.
- Answers cite their passages, and `result.documents` shows exactly what the LLM read.

## Next steps

- Understand each part: [Core concepts](concepts.md)
- Pick the right providers: [Choosing components](choosing-components.md)
- Build something complete: [Chat with a PDF](../tutorials/chat-with-pdf.md) or a
  [Streamlit app](../tutorials/streamlit-app.md)
