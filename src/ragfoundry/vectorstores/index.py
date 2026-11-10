"""Uniform add, delete and search over any LangChain vector store."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .._utils import sanitize_metadata

FilterTranslator = Callable[[dict[str, Any]], dict[str, Any]]


class VectorIndex:
    """Wrap a LangChain ``VectorStore`` and hide per-provider differences."""

    def __init__(
        self,
        store: VectorStore,
        translate_filter: FilterTranslator | None = None,
        save: Callable[[], None] | None = None,
        before_add: Callable[[Sequence[str]], None] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self._translate_filter = translate_filter or (lambda where: {"filter": where})
        self._save = save
        self._before_add = before_add
        self._close = close

    def add(self, chunks: Sequence[Document]) -> None:
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        if self._before_add:
            self._before_add(ids)
        clean = [
            Document(page_content=chunk.page_content, metadata=sanitize_metadata(chunk.metadata))
            for chunk in chunks
        ]
        self.store.add_documents(clean, ids=ids)
        self.save()

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self.store.delete(ids=list(ids))
            self.save()

    def search(self, query: str, k: int, where: dict[str, Any] | None = None) -> list[Document]:
        kwargs = self._translate_filter(where) if where else {}
        return self.store.similarity_search(query, k=k, **kwargs)

    def save(self) -> None:
        if self._save:
            self._save()

    def close(self) -> None:
        if self._close:
            self._close()
