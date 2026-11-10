"""LLM-judged generation metrics.

Each function returns a score from 0 to 1, or ``None`` when the metric does not apply or the judge
reply could not be interpreted. ``None`` is excluded from averages, never counted as zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput

from .._utils import message_text, parse_json, string_items
from ..prompts import (
    ANSWER_CORRECTNESS_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    EXTRACT_CLAIMS_PROMPT,
    GRADE_PROMPT,
    VERIFY_CLAIM_PROMPT,
)


def _yes_fraction(
    llm: BaseChatModel, prompts: list[LanguageModelInput], config: Any
) -> float | None:
    votes = []
    for reply in llm.batch(prompts, config=config):
        verdict = message_text(reply).strip().lower()
        if verdict.startswith("yes"):
            votes.append(True)
        elif verdict.startswith("no"):
            votes.append(False)
    return sum(votes) / len(votes) if votes else None


def _rubric_score(llm: BaseChatModel, prompt: str, config: Any) -> float | None:
    parsed = parse_json(message_text(llm.invoke(prompt, config=config)))
    score = parsed.get("score") if isinstance(parsed, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return max(0.0, min(1.0, float(score)))


def _claims(llm: BaseChatModel, question: str, text: str, config: Any) -> list[str] | None:
    prompt = EXTRACT_CLAIMS_PROMPT.format(question=question or "(not given)", answer=text)
    reply = llm.invoke(prompt, config=config)
    return string_items(parse_json(message_text(reply)))


def faithfulness(
    llm: BaseChatModel,
    answer: str,
    contexts: Sequence[str],
    question: str = "",
    config: Any = None,
) -> float | None:
    """Fraction of the answer's factual claims supported by the retrieved contexts."""
    if not contexts:
        return None
    claims = _claims(llm, question, answer, config)
    if not claims:
        return None
    context = "\n\n".join(contexts)
    prompts: list[LanguageModelInput] = [
        VERIFY_CLAIM_PROMPT.format(context=context, claim=claim) for claim in claims
    ]
    return _yes_fraction(llm, prompts, config)


def answer_relevance(
    llm: BaseChatModel, question: str, answer: str, config: Any = None
) -> float | None:
    """How directly the answer addresses the question."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    return _rubric_score(llm, prompt, config)


def answer_correctness(
    llm: BaseChatModel, question: str, answer: str, reference: str, config: Any = None
) -> float | None:
    """Factual agreement between the answer and a reference answer."""
    prompt = ANSWER_CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
    return _rubric_score(llm, prompt, config)


def context_relevance(
    llm: BaseChatModel, question: str, contexts: Sequence[str], config: Any = None
) -> float | None:
    """Fraction of retrieved contexts useful for the question."""
    if not contexts:
        return None
    prompts: list[LanguageModelInput] = [
        GRADE_PROMPT.format(question=question, passage=context) for context in contexts
    ]
    return _yes_fraction(llm, prompts, config)


def context_recall(
    llm: BaseChatModel,
    reference: str,
    contexts: Sequence[str],
    question: str = "",
    config: Any = None,
) -> float | None:
    """Fraction of the reference answer's claims that the retrieved contexts support."""
    if not contexts:
        return None
    claims = _claims(llm, question, reference, config)
    if not claims:
        return None
    context = "\n\n".join(contexts)
    prompts: list[LanguageModelInput] = [
        VERIFY_CLAIM_PROMPT.format(context=context, claim=claim) for claim in claims
    ]
    return _yes_fraction(llm, prompts, config)
