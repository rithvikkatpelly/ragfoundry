"""Ask a chat model where the topic boundaries in a text are."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from .._utils import message_text, parse_json
from ..prompts import LLM_CHUNK_PROMPT
from ._text import sentence_spans

logger = logging.getLogger(__name__)


class LLMChunker(TextSplitter):
    """Group sentences into topical chunks chosen by an LLM.

    Long texts are processed in windows. The final group of each window is carried into the next,
    so a topic is not cut merely because a window ended. If a reply cannot be parsed, that window
    falls back to recursive splitting and a warning is logged.
    """

    def __init__(self, llm: BaseChatModel, window_chars: int = 8000, **kwargs: Any) -> None:
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self._llm = llm
        self._window_chars = window_chars
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=self._length_function,
        )

    def split_text(self, text: str) -> list[str]:
        spans = sentence_spans(text)
        if len(spans) < 2:
            return self._bounded(text)

        chunks: list[str] = []
        start = 0
        while start < len(spans):
            end = self._window_end(spans, start)
            boundaries = self._ask(text, spans, start, end)
            if boundaries is None:
                logger.warning("Unusable LLM chunking reply; using recursive split for a window")
                chunks.extend(self._fallback.split_text(text[spans[start][0] : spans[end - 1][1]]))
                start = end
                continue
            finished = end == len(spans)
            stop = end if finished or not boundaries else boundaries[-1]
            edges = [start] + [b for b in boundaries if b < stop] + [stop]
            for first, last in zip(edges, edges[1:], strict=False):
                chunks.extend(self._bounded(text[spans[first][0] : spans[last - 1][1]]))
            start = stop
        return chunks

    def _window_end(self, spans: list[tuple[int, int]], start: int) -> int:
        end = start + 1
        while end < len(spans) and spans[end][1] - spans[start][0] <= self._window_chars:
            end += 1
        return end

    def _ask(
        self, text: str, spans: list[tuple[int, int]], start: int, end: int
    ) -> list[int] | None:
        """Return sorted sentence indices inside ``(start, end)`` that open a chunk."""
        if end - start < 2:
            return []
        numbered = "\n".join(
            f"[{index}] {text[spans[index][0] : spans[index][1]].strip()}"
            for index in range(start, end)
        )
        reply = self._llm.invoke(LLM_CHUNK_PROMPT.format(sentences=numbered))
        parsed = parse_json(message_text(reply))
        if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
            return None
        return sorted({item for item in parsed if start < item < end})

    def _bounded(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self._length_function(text) <= self._chunk_size:
            return [text]
        return self._fallback.split_text(text)
