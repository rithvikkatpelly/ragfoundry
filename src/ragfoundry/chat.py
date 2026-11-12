"""Conversations: a pipeline plus the history of one chat."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ._utils import message_text
from .context import HistoryInput, normalize_history

if TYPE_CHECKING:
    from .pipeline import RAGPipeline, RAGResult


class ChatSession:
    """Asks questions with the conversation so far, and records each question and answer.

    The session keeps the full history; ``context_window.max_history_turns`` and
    ``max_history_tokens`` decide how much of it each request sends. Create one session per
    conversation (per user, per browser tab) with :meth:`RAGPipeline.chat`. Sessions share the
    pipeline's models and connections, so they are cheap to create.
    """

    def __init__(self, pipeline: RAGPipeline, history: HistoryInput | None = None) -> None:
        self.pipeline = pipeline
        self.history: list[BaseMessage] = normalize_history(history)

    def ask(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> RAGResult:
        """Answer a question in the context of this conversation."""
        result = self.pipeline.query(question, where=where, top_k=top_k, history=self.history)
        self._record(question, result.answer)
        return result

    async def aask(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> RAGResult:
        """Async form of :meth:`ask`."""
        result = await self.pipeline.aquery(
            question, where=where, top_k=top_k, history=self.history
        )
        self._record(question, result.answer)
        return result

    def stream(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> Iterator[str]:
        """Yield the answer as it is generated; the turn is recorded once it finishes."""
        parts: list[str] = []
        for text in self.pipeline.stream(question, where=where, top_k=top_k, history=self.history):
            parts.append(text)
            yield text
        self._record(question, "".join(parts))

    async def astream(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> AsyncIterator[str]:
        """Async form of :meth:`stream`."""
        parts: list[str] = []
        async for text in self.pipeline.astream(
            question, where=where, top_k=top_k, history=self.history
        ):
            parts.append(text)
            yield text
        self._record(question, "".join(parts))

    def reset(self) -> None:
        """Forget the conversation."""
        self.history.clear()

    def to_dicts(self) -> list[dict[str, str]]:
        """The history as ``{"role": "user" | "assistant", "content": ...}`` dicts for JSON."""
        roles = {"human": "user", "ai": "assistant"}
        return [{"role": roles[m.type], "content": message_text(m)} for m in self.history]

    def _record(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

    def __len__(self) -> int:
        """Number of completed question-and-answer turns."""
        return sum(1 for message in self.history if message.type == "human")
