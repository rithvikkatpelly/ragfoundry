# Custom components

ragfoundry is built on LangChain, so any LangChain model or vector store can replace the
configured one, and you can register new file types and chunking strategies.

## Bring your own LangChain objects

```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

rag = RAGPipeline(
    config,
    llm=AzureChatOpenAI(azure_deployment="gpt-4o", api_version="2024-10-21"),
    embeddings=AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small"),
)
```

| Argument | Accepts | Replaces |
|---|---|---|
| `llm` | Any LangChain `BaseChatModel` | `llm` section |
| `embeddings` | Any LangChain `Embeddings` | `embeddings` section |
| `vector_store` | Any LangChain `VectorStore` | `vector_store` section |
| `reranker` | Any object with `rerank(query, documents, top_n) -> list[Document]` | `retrieval.rerank` |
| `token_counter` | Any function `str -> int` | `context_window.token_counter` |

This is how you use providers ragfoundry doesn't configure directly: Azure OpenAI, AWS Bedrock,
Mistral, Together, local llama.cpp, Elasticsearch, OpenSearch, and so on.

!!! note "Filters with your own vector store"
    `where` is passed to the store as `filter=where`. That works for stores that accept a dict of
    exact matches; check your store's LangChain docs.

## Add a file type

Register a function that returns a LangChain document loader for the extension:

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from ragfoundry import register_loader


@register_loader(".pptx", ".ppt")
def powerpoint(path, loading_config):
    return UnstructuredPowerPointLoader(str(path))


rag.ingest("slides")          # .pptx files are now picked up
```

To add your own metadata, wrap a loader:

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DepartmentLoader(BaseLoader):
    def __init__(self, path):
        self.path = path

    def lazy_load(self):
        department = self.path.parent.name            # docs/finance/x.txt -> "finance"
        yield Document(page_content=self.path.read_text(),
                       metadata={"source": str(self.path), "department": department})


@register_loader(".txt")
def txt_with_department(path, loading_config):
    return DepartmentLoader(path)
```

Registering an extension that already has a loader replaces it.

## Add a chunking strategy

Register a function that returns a LangChain `TextSplitter`:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, TextSplitter
from ragfoundry import register_chunker


class HeadingSplitter(TextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._headers = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2")])

    def split_text(self, text):
        return [section.page_content for section in self._headers.split_text(text)]


@register_chunker("by_heading")
def by_heading(chunking_config, resources):
    return HeadingSplitter(chunk_size=chunking_config.chunk_size, chunk_overlap=0)
```

```python
"chunking": {"strategy": "by_heading"}
```

`resources.get_embeddings()` and `resources.get_llm()` give your splitter the pipeline's models
if it needs them.

## Use the building blocks directly

```python
from ragfoundry import chunk_documents, load_sources
from ragfoundry.config import ChunkingConfig, LoadingConfig

documents = load_sources("docs", LoadingConfig()).documents
chunks = chunk_documents(documents, ChunkingConfig(chunk_size=500, chunk_overlap=50))
```
