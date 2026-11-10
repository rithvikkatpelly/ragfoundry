"""Vector and BM25 candidate retrieval."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..chunk_store import ChunkStore
from ..config import RetrievalConfig
from ..vectorstores import VectorIndex
from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 over the chunk store, rebuilt when the store's revision changes.

    The index is held in process memory and each search scores every chunk, which suits corpora
    up to roughly a few hundred thousand chunks.
    """

    def __init__(self, chunk_store: ChunkStore) -> None:
        self._store = chunk_store
        self._lock = threading.Lock()
        self._revision = -1
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def _refresh(self) -> tuple[BM25Okapi | None, list[Document]]:
        with self._lock:
            revision = self._store.revision()
            if revision != self._revision:
                self._documents = self._store.documents()
                corpus = [_tokenize(doc.page_content) for doc in self._documents]
                self._bm25 = BM25Okapi(corpus) if corpus else None
                self._revision = revision
                if not corpus:
                    logger.warning("Hybrid search enabled but no chunks have been ingested here")
            return self._bm25, self._documents

    def search(self, query: str, k: int, where: dict[str, Any] | None = None) -> list[Document]:
        bm25, documents = self._refresh()
        tokens = _tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        # Rank by whether a query term occurs, since BM25 IDF can be zero in small corpora.
        query_terms = set(tokens)
        ranked: list[Document] = []
        for index in np.argsort(-scores, kind="stable"):
            document = documents[int(index)]
            if where and any(document.metadata.get(a) != b for a, b in where.items()):
                continue
            if query_terms.isdisjoint(_tokenize(document.page_content)):
                continue
            ranked.append(document)
            if len(ranked) == k:
                break
        return ranked


class Retriever:
    """Run each query against the vector index and, if enabled, BM25, then fuse the rankings."""

    def __init__(
        self, index: VectorIndex, config: RetrievalConfig, bm25: BM25Index | None = None
    ) -> None:
        self._index = index
        self._config = config
        self._bm25 = bm25

    def retrieve(
        self,
        queries: Sequence[str],
        vector_only_queries: Sequence[str] = (),
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        hybrid = self._config.hybrid
        fetch_k = limit or self._config.fetch_k
        rankings: list[tuple[list[Document], float]] = []
        for query in [*queries, *vector_only_queries]:
            rankings.append((self._index.search(query, fetch_k, where), hybrid.vector_weight))
        if hybrid.enabled and self._bm25 is not None:
            for query in queries:
                rankings.append((self._bm25.search(query, fetch_k, where), hybrid.bm25_weight))
        return reciprocal_rank_fusion(rankings, hybrid.rrf_k)
