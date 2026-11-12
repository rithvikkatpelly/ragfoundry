"""Query workflow: condense, transform, retrieve, rerank, grade and generate."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .._utils import message_text
from ..config import RAGConfig
from ..context import (
    TokenCounter,
    approximate_tokens,
    format_context,
    pack_documents,
    render_history,
    window_history,
)
from ..prompts import (
    CONDENSE_QUESTION_PROMPT,
    CORRECTIVE_REWRITE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_PROMPT,
    GRADE_PROMPT,
)
from ..retrieval import Reranker, Retriever, transform_query
from ..retrieval.fusion import doc_key

logger = logging.getLogger(__name__)


class QueryState(TypedDict, total=False):
    question: str
    history: list[BaseMessage]
    search_question: str
    where: dict[str, Any] | None
    top_k: int | None
    queries: list[str]
    vector_only: list[str]
    all_queries: list[str]
    documents: list[Document]
    relevant: list[Document]
    retries: int
    answer: str
    abstained: bool
    context_tokens: int


@dataclass
class QueryComponents:
    config: RAGConfig
    retriever: Retriever
    llm: Callable[[], BaseChatModel] | None = None
    reranker: Reranker | None = None
    token_counter: TokenCounter = field(default=approximate_tokens)


def build_query_graph(components: QueryComponents, generate_answer: bool = True) -> Any:
    """Compile a graph holding only the stages enabled by the configuration.

    With ``generate_answer=False`` the graph stops after retrieval, which is what retrieval-only
    calls and retrieval evaluation use.
    """
    config = components.config
    settings = config.retrieval
    window = config.context_window
    count = components.token_counter
    corrective = settings.corrective.enabled
    transforms = bool(settings.query_transform.strategies)

    def need_llm() -> BaseChatModel:
        if components.llm is None:
            raise ValueError("This pipeline step requires an LLM")
        return components.llm()

    def limit(state: QueryState) -> int:
        return state.get("top_k") or settings.top_k

    def prepare(state: QueryState) -> QueryState:
        question = state["question"]
        history = window_history(
            state.get("history") or [], window.max_history_turns, window.max_history_tokens, count
        )
        return {
            "history": history,
            "search_question": question,
            "queries": [question],
            "vector_only": [],
            "all_queries": [question],
            "retries": 0,
        }

    def condense(state: QueryState, config: RunnableConfig) -> QueryState:
        if not state["history"]:
            return {}
        prompt = CONDENSE_QUESTION_PROMPT.format(
            history=render_history(state["history"]), question=state["question"]
        )
        standalone = message_text(need_llm().invoke(prompt, config=config)) or state["question"]
        return {"search_question": standalone, "queries": [standalone], "all_queries": [standalone]}

    def transform(state: QueryState, config: RunnableConfig) -> QueryState:
        result = transform_query(
            state["search_question"], need_llm(), settings.query_transform, config
        )
        return {
            "queries": result.queries,
            "vector_only": result.vector_only,
            "all_queries": result.queries + result.vector_only,
        }

    def retrieve(state: QueryState) -> QueryState:
        wanted = limit(state)
        candidates = components.retriever.retrieve(
            state["queries"],
            state.get("vector_only", []),
            state.get("where"),
            limit=max(settings.fetch_k, wanted),
        )
        keep = candidates if components.reranker else candidates[:wanted]
        return {"documents": keep}

    def rerank(state: QueryState) -> QueryState:
        assert components.reranker is not None
        return {
            "documents": components.reranker.rerank(
                state["search_question"], state["documents"], limit(state)
            )
        }

    def grade(state: QueryState, config: RunnableConfig) -> QueryState:
        documents = state["documents"]
        kept = {doc_key(doc): doc for doc in state.get("relevant", [])}
        if documents:
            prompts: list[LanguageModelInput] = [
                GRADE_PROMPT.format(question=state["search_question"], passage=doc.page_content)
                for doc in documents
            ]
            for doc, reply in zip(documents, need_llm().batch(prompts, config=config), strict=True):
                verdict = message_text(reply).strip().lower()
                if not verdict.startswith(("yes", "no")):
                    logger.warning("Unclear relevance grade %r; keeping document", verdict[:40])
                if not verdict.startswith("no"):
                    kept.setdefault(doc_key(doc), doc)
        return {"relevant": list(kept.values()), "documents": list(kept.values())}

    def route_after_grade(state: QueryState) -> str:
        enough = len(state["relevant"]) >= settings.corrective.min_relevant_docs
        if not enough and state["retries"] < settings.corrective.max_retries:
            return "rewrite"
        if not generate_answer:
            return END
        return "generate" if state["relevant"] else "abstain"

    def rewrite(state: QueryState, config: RunnableConfig) -> QueryState:
        prompt = CORRECTIVE_REWRITE_PROMPT.format(
            question=state["search_question"], previous="; ".join(state["all_queries"])
        )
        query = message_text(need_llm().invoke(prompt, config=config)) or state["search_question"]
        return {
            "queries": [query],
            "vector_only": [],
            "all_queries": [*state["all_queries"], query],
            "retries": state["retries"] + 1,
        }

    def route_after_retrieval(state: QueryState) -> str:
        if not generate_answer:
            return END
        return "generate" if state["documents"] else "abstain"

    def generate(state: QueryState, config: RunnableConfig) -> QueryState:
        system = components.config.generation.system_prompt or DEFAULT_SYSTEM_PROMPT
        packed = pack_documents(
            state["documents"][: limit(state)],
            window.max_context_tokens,
            count,
            truncate_last=window.truncate_last_passage,
        )
        if packed.dropped:
            logger.debug("Context budget left out %d passage(s)", packed.dropped)
        human = GENERATION_PROMPT.format(
            context=format_context(packed.documents), question=state["question"]
        )
        messages = [SystemMessage(content=system), *state["history"], HumanMessage(content=human)]
        reply = need_llm().invoke(messages, config=config)
        return {
            "answer": message_text(reply),
            "documents": packed.documents,
            "abstained": False,
            "context_tokens": packed.tokens,
        }

    def abstain(state: QueryState) -> QueryState:
        return {"answer": components.config.generation.abstain_message, "abstained": True}

    graph = StateGraph(QueryState)
    graph.add_node("prepare", prepare)
    graph.add_node("retrieve", retrieve)
    graph.add_edge(START, "prepare")

    before_retrieval = "prepare"
    if window.condense_question and window.max_history_turns > 0:
        graph.add_node("condense", condense)
        graph.add_edge("prepare", "condense")
        before_retrieval = "condense"
    if transforms:
        graph.add_node("transform", transform)
        graph.add_edge(before_retrieval, "transform")
        graph.add_edge("transform", "retrieve")
    else:
        graph.add_edge(before_retrieval, "retrieve")

    last_stage = "retrieve"
    if components.reranker:
        graph.add_node("rerank", rerank)
        graph.add_edge("retrieve", "rerank")
        last_stage = "rerank"

    if generate_answer:
        graph.add_node("generate", generate)
        graph.add_node("abstain", abstain)
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)

    targets = ["generate", "abstain"] if generate_answer else [END]
    if corrective:
        graph.add_node("grade", grade)
        graph.add_node("rewrite", rewrite)
        graph.add_edge(last_stage, "grade")
        graph.add_conditional_edges("grade", route_after_grade, ["rewrite", *targets])
        graph.add_edge("rewrite", "retrieve")
    else:
        graph.add_conditional_edges(last_stage, route_after_retrieval, targets)
    return graph.compile()
