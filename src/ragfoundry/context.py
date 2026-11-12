"""Context window management: token counting, passage packing and chat history windows."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ._utils import message_text, require
from .config import ContextWindowConfig
from .errors import ConfigurationError

__all__ = [
    "HistoryInput",
    "PackedContext",
    "TokenCounter",
    "build_token_counter",
    "format_block",
    "format_context",
    "normalize_history",
    "pack_documents",
    "window_history",
]

TokenCounter = Callable[[str], int]
"""Returns the number of tokens in a string."""

HistoryInput = Iterable[BaseMessage | Mapping[str, Any] | tuple[str, str]]
"""Chat history: LangChain messages, ``{"role", "content"}`` dicts or ``(role, content)`` pairs."""

# Passages shorter than this after trimming add little and are left out instead.
_MIN_TRIMMED_TOKENS = 16
_SEPARATOR = "\n\n"


def build_token_counter(config: ContextWindowConfig) -> TokenCounter:
    """Create the counter named by ``token_counter``: ``approximate`` or ``tiktoken``."""
    if config.token_counter == "approximate":
        return approximate_tokens
    tiktoken = require("tiktoken", "tokens")
    encoding = tiktoken.get_encoding(config.encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def approximate_tokens(text: str) -> int:
    """About four characters per token, which holds for English text with most tokenizers."""
    return math.ceil(len(text) / 4)


# ------------------------------------------------------------------------ passages
def format_block(number: int, document: Document) -> str:
    source = document.metadata.get("source", "unknown")
    return f"[{number}] (source: {source})\n{document.page_content}"


def format_context(documents: Sequence[Document]) -> str:
    return _SEPARATOR.join(
        format_block(number, document) for number, document in enumerate(documents, start=1)
    )


@dataclass
class PackedContext:
    documents: list[Document]
    tokens: int
    dropped: int = 0
    truncated: bool = False


def pack_documents(
    documents: Sequence[Document],
    max_tokens: int | None,
    count: TokenCounter,
    *,
    truncate_last: bool = True,
) -> PackedContext:
    """Keep passages in rank order until the token budget is used.

    The passage that would cross the budget is shortened to fit when ``truncate_last`` is set, so
    the budget is filled rather than left partly empty. A shortened passage is a copy with
    ``metadata["truncated"] = True``; the originals are never modified.
    """
    kept: list[Document] = []
    used = 0
    truncated = False
    for document in documents:
        separator = count(_SEPARATOR) if kept else 0
        tokens = count(format_block(len(kept) + 1, document)) + separator
        if max_tokens is None or used + tokens <= max_tokens:
            kept.append(document)
            used += tokens
            continue
        remaining = max_tokens - used - separator
        if truncate_last and remaining >= _MIN_TRIMMED_TOKENS:
            trimmed = _trim_to_fit(document, len(kept) + 1, remaining, count)
            if trimmed is not None:
                kept.append(trimmed)
                used += count(format_block(len(kept), trimmed)) + separator
                truncated = True
        break
    return PackedContext(kept, used, dropped=len(documents) - len(kept), truncated=truncated)


def _trim_to_fit(
    document: Document, number: int, budget: int, count: TokenCounter
) -> Document | None:
    """Return the longest prefix of the passage whose formatted block fits ``budget`` tokens."""
    text = document.page_content
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _copy(document, text[:middle].rstrip() + " …")
        if count(format_block(number, candidate)) <= budget:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        return None
    prefix = text[:low]
    # Prefer ending on a word boundary when one is close.
    cut = prefix.rfind(" ")
    if cut > low * 0.8:
        prefix = prefix[:cut]
    return _copy(document, prefix.rstrip() + " …")


def _copy(document: Document, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={**document.metadata, "truncated": True},
        id=document.id,
    )


# ------------------------------------------------------------------------- history
_ROLES = {
    "human": HumanMessage,
    "user": HumanMessage,
    "ai": AIMessage,
    "assistant": AIMessage,
}


def normalize_history(history: HistoryInput | None) -> list[BaseMessage]:
    """Convert supported history formats to LangChain messages, keeping only user and AI turns."""
    if history is None:
        return []
    messages: list[BaseMessage] = []
    for item in history:
        if isinstance(item, BaseMessage):
            message = item
        else:
            if isinstance(item, Mapping):
                role, content = item.get("role"), item.get("content")
            elif isinstance(item, tuple) and len(item) == 2:
                role, content = item
            else:
                raise ConfigurationError(
                    f"Unsupported history item {item!r}: use a message, a dict with 'role' and "
                    "'content', or a (role, content) tuple"
                )
            kind = _ROLES.get(str(role).lower())
            if kind is None:
                raise ConfigurationError(
                    f"Unknown history role {role!r}: use 'user'/'human' or 'assistant'/'ai'"
                )
            message = kind(content=str(content))
        if message.type in ("human", "ai"):
            messages.append(message)
    return messages


def window_history(
    messages: Sequence[BaseMessage],
    max_turns: int,
    max_tokens: int | None,
    count: TokenCounter,
) -> list[BaseMessage]:
    """Keep the most recent ``max_turns`` user messages with their replies, within ``max_tokens``.

    A turn is a user message plus the replies after it. Whole turns are dropped, oldest first, so
    the model never sees a reply without the question that prompted it.
    """
    if max_turns == 0 or not messages:
        return []
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if message.type == "human" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    turns = turns[-max_turns:]
    if max_tokens is not None:
        total = 0
        kept: list[list[BaseMessage]] = []
        for turn in reversed(turns):
            size = sum(count(message_text(message)) for message in turn)
            if total + size > max_tokens:
                break
            kept.append(turn)
            total += size
        turns = list(reversed(kept))
    return [message for turn in turns for message in turn]


def render_history(messages: Sequence[BaseMessage]) -> str:
    """Plain-text transcript used in the question-condensing prompt."""
    labels = {"human": "User", "ai": "Assistant"}
    return "\n".join(f"{labels[m.type]}: {message_text(m)}" for m in messages)
