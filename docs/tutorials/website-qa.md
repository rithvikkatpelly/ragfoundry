# Question answering over a website

Crawl a documentation site, index its pages and answer questions about them, linking each answer
to the page it came from.

**Time:** 15 minutes · **Needs:** a [Groq API key](../get-started/api-keys.md).

## 1. Install

```bash
pip install "ragfoundry[web,huggingface,chroma,groq]"
```

## 2. Configure crawling

```python title="site_qa.py"
from ragfoundry import RAGConfig, RAGPipeline

config = RAGConfig.model_validate({
    "embeddings": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "vector_store": {"provider": "chroma", "collection_name": "site"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0},
    "loading": {
        "on_error": "skip",
        "web": {
            "max_depth": 2,          # the start page, pages it links to, and pages they link to
            "max_pages": 100,        # stop after this many pages
            "same_domain": True,     # never leave the site
            "respect_robots": True,  # obey the site's robots.txt
            "delay_seconds": 0.5,    # be polite: wait between requests
        },
    },
    "retrieval": {"top_k": 4, "hybrid": {"enabled": True}},
})
```

!!! warning "Crawl responsibly"
    Only crawl sites you are allowed to. Keep `respect_robots: true` and a delay between requests.

## 3. Index and ask

```python
with RAGPipeline(config) as rag:
    report = rag.ingest("https://docs.python.org/3/tutorial/")
    print(f"Fetched {report.documents_loaded} pages, {report.chunks_total} chunks")

    result = rag.query("How do I create a virtual environment?")
    print(result.answer)
    for doc in result.documents:
        print(" -", doc.metadata.get("title"), doc.metadata["source"])
```

Each page becomes a document with its URL as `source` and its HTML title as `title`.
Navigation, headers, footers, scripts and forms are removed, so only the main text is indexed.

## 4. Keep it up to date

Run the same `ingest` call on a schedule (a daily cron job, for example). Unchanged pages are
skipped; changed pages are re-embedded. A page removed from the site stays indexed until you
remove it:

```python
rag.delete_source("https://docs.python.org/3/tutorial/old-page.html")
```

## Mixing sources

`ingest` takes a list, so a site, local files and single pages can share one index:

```python
rag.ingest(["https://example.com/docs/", "./internal-notes", "./faq.pdf"])
```

## Next

[Web loader reference](../integrations/loaders/web.md) ·
[Keep the index up to date](../how-to/managing-the-index.md)
