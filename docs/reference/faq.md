# FAQ

**Do I need to know LangChain?**
No. ragfoundry uses LangChain internally, but you only write a configuration. If you know
LangChain, you can pass in any LangChain model or vector store.

**Is it free?**
ragfoundry is free and open source (MIT). Hugging Face embeddings, local vector stores and Ollama
are free. Hosted providers charge for their APIs; Groq and Gemini have free tiers.

**Can my documents stay private?**
Yes. With Hugging Face embeddings, a local vector store (Chroma, FAISS, Qdrant local, Milvus Lite)
and Ollama, nothing leaves your machine. See [Fully local RAG](../tutorials/local-rag.md).

**How many documents can it handle?**
Local stores comfortably handle hundreds of thousands of chunks. For millions, use a server store
such as Qdrant, Milvus, Pinecone or PGVector. Hybrid search keeps its keyword index in memory, which
suits up to a few hundred thousand chunks.

**Which languages are supported?**
Any language the embedding model and LLM support. For non-English documents, use a multilingual
embedding model such as `intfloat/multilingual-e5-base` or OpenAI's `text-embedding-3-small`.

**Can it answer questions about tables and spreadsheets?**
It can look up rows ("what does part X cost?"). It can't total or average a whole table, because
it only sees a few rows per question. Use SQL or pandas for that.

**Can it read images or scanned documents?**
Not directly. Run OCR on scanned PDFs first. Images inside documents are ignored.

**How do I update documents?**
Run `ingest` again; only changes are processed. Use `delete_source` for removed documents. See
[Keep the index up to date](../how-to/managing-the-index.md).

**Can several users chat at once?**
Yes. Create one pipeline and one `rag.chat()` session per user. See the
[FastAPI tutorial](../tutorials/fastapi-chatbot.md).

**Does it work in Jupyter and Google Colab?**
Yes. See the [Colab notebook](https://colab.research.google.com/github/rithvikkatpelly/ragfoundry/blob/main/examples/colab_quickstart.ipynb).

**Is there a command-line tool?**
Not since 0.2.0. Use the Python API, with YAML via `RAGPipeline.from_yaml()` if you prefer
configuration files.

**How do I cite pages in answers?**
`result.documents` holds each cited passage with its metadata; PDFs include `page`. See
[PDF](../integrations/loaders/pdf.md#show-page-numbers-in-citations).
