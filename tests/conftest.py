"""Deterministic stand-ins for embedding and chat models."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_STOP = {"the", "a", "an", "is", "of", "and", "to", "in", "what", "how", "does", "do", "it"}


class KeywordEmbeddings(Embeddings):
    """Hashed bag of words: texts sharing content words are close in cosine distance."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension)
        for word in re.findall(r"\w+", text.lower()):
            if word not in _STOP:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._embed(text)


class RuleChatModel(BaseChatModel):
    """Replies with ``responder(prompt_text)`` and records every prompt it receives."""

    responder: Callable[[str], str]
    prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "rule"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "\n".join(str(message.content) for message in messages)
        self.prompts.append(text)
        reply = self.responder(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._generate(messages).generations[0].message.content
        for token in re.findall(r"\S+\s*", str(reply)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk


def rule_llm(responder: Callable[[str], str]) -> RuleChatModel:
    return RuleChatModel(responder=responder, prompts=[])


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def corpus(tmp_path):
    """A small document set covering distinct topics."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "solar.md").write_text(
        "# Solar power\n\nSolar panels convert sunlight into electricity using photovoltaic "
        "cells. Panel efficiency for silicon cells is typically around 20 percent.\n",
        encoding="utf-8",
    )
    (docs / "baking.txt").write_text(
        "Sourdough bread uses a fermented starter instead of commercial yeast. The starter "
        "needs regular feeding with flour and water.\n",
        encoding="utf-8",
    )
    (docs / "parts.csv").write_text(
        "part_number,description\nXQ-4471,hydraulic pump gasket\nZT-9902,brake caliper\n",
        encoding="utf-8",
    )
    return docs
