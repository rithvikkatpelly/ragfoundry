"""Chunker registry and stable chunk identity."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from .._utils import require
from ..config import ChunkingConfig
from ..errors import ConfigurationError
from .llm import LLMChunker
from .semantic import SemanticChunker

_NAMESPACE = uuid.UUID("5f0c1e0a-7c55-4d0b-9c55-6f1f3f0d9a11")


@dataclass
class ChunkerResources:
    """Models a strategy may need; resolved only by the strategies that use them."""

    embeddings: Callable[[], Embeddings] | None = None
    llm: Callable[[], BaseChatModel] | None = None

    def get_embeddings(self) -> Embeddings:
        if self.embeddings is None:
            raise ConfigurationError("The 'semantic' strategy requires embeddings")
        return self.embeddings()

    def get_llm(self) -> BaseChatModel:
        if self.llm is None:
            raise ConfigurationError("The 'llm' strategy requires an LLM")
        return self.llm()


class FixedWindowSplitter(TextSplitter):
    """Exact sliding windows of ``chunk_size`` characters overlapping by ``chunk_overlap``."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["strip_whitespace"] = False
        super().__init__(**kwargs)

    def split_text(self, text: str) -> list[str]:
        step = self._chunk_size - self._chunk_overlap
        windows = (text[i : i + self._chunk_size] for i in range(0, len(text), step))
        chunks = [window for window in windows if window.strip()]
        # Drop a final window that lies entirely inside the previous one.
        if len(chunks) > 1 and len(chunks[-1]) <= self._chunk_overlap:
            chunks.pop()
        return chunks


ChunkerFactory = Callable[[ChunkingConfig, ChunkerResources], TextSplitter]
_CHUNKERS: dict[str, ChunkerFactory] = {}


def register_chunker(name: str) -> Callable[[ChunkerFactory], ChunkerFactory]:
    """Register a strategy selectable through ``chunking.strategy``."""

    def decorator(factory: ChunkerFactory) -> ChunkerFactory:
        _CHUNKERS[name] = factory
        return factory

    return decorator


def _length_function(config: ChunkingConfig) -> Callable[[str], int]:
    if config.length_unit == "characters":
        return len
    tiktoken = require("tiktoken", "tokens")
    encoding = tiktoken.get_encoding(config.encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


@register_chunker("fixed")
def _fixed(config: ChunkingConfig, resources: ChunkerResources) -> TextSplitter:
    if config.length_unit == "tokens":
        require("tiktoken", "tokens")
        return TokenTextSplitter(
            encoding_name=config.encoding_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
    return FixedWindowSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, add_start_index=True
    )


@register_chunker("recursive")
def _recursive(config: ChunkingConfig, resources: ChunkerResources) -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=_length_function(config),
        separators=config.separators,
        add_start_index=True,
    )


@register_chunker("semantic")
def _semantic(config: ChunkingConfig, resources: ChunkerResources) -> TextSplitter:
    return SemanticChunker(
        resources.get_embeddings(),
        breakpoint_percentile=config.semantic.breakpoint_percentile,
        buffer_size=config.semantic.buffer_size,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=_length_function(config),
    )


@register_chunker("llm")
def _llm(config: ChunkingConfig, resources: ChunkerResources) -> TextSplitter:
    return LLMChunker(
        resources.get_llm(),
        window_chars=config.llm.window_chars,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=_length_function(config),
    )


def build_splitter(
    config: ChunkingConfig, resources: ChunkerResources | None = None
) -> TextSplitter:
    factory = _CHUNKERS.get(config.strategy)
    if factory is None:
        raise ConfigurationError(
            f"Unknown chunking strategy '{config.strategy}'. "
            f"Available: {', '.join(sorted(_CHUNKERS))}"
        )
    return factory(config, resources or ChunkerResources())


def assign_chunk_ids(chunks: Sequence[Document]) -> list[Document]:
    """Give each chunk an id derived from its source and text.

    An unchanged chunk keeps its id across runs even if earlier text in the file changes, so
    re-ingestion only embeds what is new. The occurrence count separates identical chunks.
    """
    occurrences: Counter[tuple[str, str]] = Counter()
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        occurrences[(source, digest)] += 1
        key = f"{source}\0{digest}\0{occurrences[(source, digest)]}"
        chunk.metadata["chunk_id"] = str(uuid.uuid5(_NAMESPACE, key))
    return list(chunks)


def chunk_documents(
    documents: Sequence[Document],
    config: ChunkingConfig | None = None,
    resources: ChunkerResources | None = None,
) -> list[Document]:
    """Split documents with the configured strategy and assign stable ids."""
    splitter = build_splitter(config or ChunkingConfig(), resources)
    return assign_chunk_ids(splitter.split_documents(list(documents)))
