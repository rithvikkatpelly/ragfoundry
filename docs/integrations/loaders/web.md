# Web pages

Fetches web pages, removes navigation and other boilerplate, and indexes the main text. It can
also crawl a site by following links.

| | |
|---|---|
| Sources | `http://` and `https://` URLs |
| Extra | `web` |
| One document per | page |
| Metadata | `source` (the URL), `title`, `file_type: web` |

## Install

```bash
pip install "ragfoundry[web]"
```

## Fetch one page

```python
rag.ingest("https://example.com/pricing")
```

## Crawl a site

```python
config = RAGConfig.model_validate({
    ...,
    "loading": {
        "web": {
            "max_depth": 2,
            "max_pages": 200,
        },
    },
})
rag.ingest("https://docs.example.com/")
```

## Options

Under `loading.web`:

| Setting | Default | Meaning |
|---|---|---|
| `max_depth` | `0` | Link levels to follow. 0 fetches only the given URL. |
| `max_pages` | `50` | Stop after this many pages |
| `same_domain` | `true` | Only follow links on the starting site |
| `respect_robots` | `true` | Obey the site's `robots.txt` |
| `delay_seconds` | `0.5` | Pause between requests |
| `timeout_seconds` | `30` | Give up on a page after this long |
| `max_bytes` | `10000000` | Skip pages larger than this |
| `user_agent` | `ragfoundry/0.2` | Sent with every request |

## What gets indexed

- `script`, `style`, `nav`, `header`, `footer`, `aside` and `form` elements are removed.
- Only HTML and plain-text responses are read; images, PDFs and other files linked from pages are
  skipped. To index a PDF on the web, download it and ingest the file.
- If the start URL fails or is disallowed by `robots.txt`, ingest raises an error. A failing linked
  page is logged and skipped, so one dead link doesn't stop a crawl.

## Limitations

- Pages that build their content with JavaScript (many single-page apps) may come back nearly
  empty, because scripts aren't run.
- Pages that disappear from a site stay indexed until you call `rag.delete_source(url)`.
- Only crawl sites you're allowed to, and keep `respect_robots` on.
