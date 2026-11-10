"""Reciprocal rank fusion of ranked document lists."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def doc_key(document: Document) -> str:
    return str(document.metadata.get("chunk_id") or document.page_content)


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[Sequence[Document], float]], rrf_k: int = 60
) -> list[Document]:
    """Merge ``(ranking, weight)`` pairs, best first, recording ``rrf_score`` in metadata."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking, weight in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = doc_key(document)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            documents.setdefault(key, document)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        Document(
            page_content=documents[key].page_content,
            metadata={**documents[key].metadata, "rrf_score": scores[key]},
        )
        for key in ordered
    ]
