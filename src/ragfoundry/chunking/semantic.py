"""Split text where the embedding similarity of neighbouring sentences drops."""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from ._text import sentence_spans


class SemanticChunker(TextSplitter):
    """Break at the largest semantic shifts; ``chunk_size`` is enforced as an upper bound."""

    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_percentile: float = 95,
        buffer_size: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self._embeddings = embeddings
        self._percentile = breakpoint_percentile
        self._buffer = buffer_size
        self._oversize = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=self._length_function,
        )

    def split_text(self, text: str) -> list[str]:
        spans = sentence_spans(text)
        if len(spans) < 3:
            return self._bounded(text)

        windows = [
            text[
                spans[max(0, i - self._buffer)][0] : spans[min(len(spans) - 1, i + self._buffer)][1]
            ]
            for i in range(len(spans))
        ]
        vectors = np.asarray(self._embeddings.embed_documents(windows), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = np.percentile(distances, self._percentile)

        chunks: list[str] = []
        first = 0
        for index, distance in enumerate(distances):
            if distance >= threshold and distance > 0:
                chunks.extend(self._bounded(text[spans[first][0] : spans[index][1]]))
                first = index + 1
        chunks.extend(self._bounded(text[spans[first][0] : spans[-1][1]]))
        return chunks

    def _bounded(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self._length_function(text) <= self._chunk_size:
            return [text]
        return self._oversize.split_text(text)
