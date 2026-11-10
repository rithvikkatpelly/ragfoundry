"""LLM-based query rewriting, expansion and HyDE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from .._utils import message_text, parse_json, string_items
from ..config import QueryTransformConfig
from ..prompts import HYDE_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT


@dataclass
class TransformedQuery:
    """``queries`` go to all retrievers; hypothetical passages go to vector search only."""

    queries: list[str]
    vector_only: list[str] = field(default_factory=list)


def transform_query(
    question: str,
    llm: BaseChatModel,
    config: QueryTransformConfig,
    run_config: Any = None,
) -> TransformedQuery:
    def ask(prompt: str) -> str:
        return message_text(llm.invoke(prompt, config=run_config))

    result = TransformedQuery(queries=[question])
    for strategy in config.strategies:
        if strategy == "rewrite":
            rewritten = ask(REWRITE_PROMPT.format(question=question))
            if rewritten:
                result.queries[0] = rewritten
        elif strategy == "multi_query":
            reply = ask(MULTI_QUERY_PROMPT.format(count=config.num_queries, question=question))
            result.queries.extend(_query_list(reply)[: config.num_queries])
        elif strategy == "hyde":
            passage = ask(HYDE_PROMPT.format(question=question))
            if passage:
                result.vector_only.append(passage)
    result.queries = list(dict.fromkeys(result.queries))
    return result


def _query_list(reply: str) -> list[str]:
    parsed = string_items(parse_json(reply))
    if parsed is not None:
        return parsed
    return [line.strip(" -*\t") for line in reply.splitlines() if line.strip(" -*\t")]
