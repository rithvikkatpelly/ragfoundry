"""Second-stage reranking of retrieved candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from .._utils import require
from ..config import RerankConfig


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[Document], top_n: int) -> list[Document]: ...


def _scored(document: Document, score: float) -> Document:
    return Document(
        page_content=document.page_content,
        metadata={**document.metadata, "rerank_score": float(score)},
    )


class CrossEncoderReranker:
    """Local cross-encoder from Hugging Face."""

    def __init__(self, model: str) -> None:
        module = require("sentence_transformers", "huggingface")
        self._model = module.CrossEncoder(model)

    def rerank(self, query: str, documents: Sequence[Document], top_n: int) -> list[Document]:
        if not documents:
            return []
        scores = self._model.predict([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [_scored(doc, score) for doc, score in ranked[:top_n]]


class CohereReranker:
    """Cohere's hosted rerank API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        module = require("langchain_cohere", "cohere")
        credentials = {"cohere_api_key": api_key} if api_key else {}
        self._client = module.CohereRerank(model=model, **credentials)

    def rerank(self, query: str, documents: Sequence[Document], top_n: int) -> list[Document]:
        if not documents:
            return []
        results = self._client.rerank(
            documents=[doc.page_content for doc in documents], query=query, top_n=top_n
        )
        return [_scored(documents[r["index"]], r["relevance_score"]) for r in results]


def build_reranker(config: RerankConfig) -> Reranker:
    if config.provider == "huggingface":
        return CrossEncoderReranker(config.model)
    key = config.api_key.get_secret_value() if config.api_key else None
    return CohereReranker(config.model, key)
