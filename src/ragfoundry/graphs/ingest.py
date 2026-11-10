"""Ingestion workflow: load, chunk, then synchronise the index."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from ..chunk_store import ChunkStore
from ..chunking import assign_chunk_ids
from ..config import RAGConfig
from ..loaders import LoadFailure, load_sources
from ..vectorstores import VectorIndex


@dataclass
class IngestReport:
    documents_loaded: int = 0
    chunks_total: int = 0
    chunks_added: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    failures: list[LoadFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestState(TypedDict, total=False):
    sources: list[str]
    documents: list[Document]
    chunks: list[Document]
    report: IngestReport


def build_ingest_graph(
    config: RAGConfig,
    make_splitter: Callable[[], Any],
    index: Callable[[], VectorIndex],
    chunk_store: Callable[[], ChunkStore],
) -> Any:
    def load(state: IngestState) -> IngestState:
        result = load_sources(state["sources"], config.loading)
        report = IngestReport(documents_loaded=len(result.documents), failures=result.failures)
        return {"documents": result.documents, "report": report}

    def chunk(state: IngestState) -> IngestState:
        pieces = make_splitter().split_documents(state["documents"])
        return {"chunks": assign_chunk_ids([c for c in pieces if c.page_content.strip()])}

    def synchronise(state: IngestState) -> IngestState:
        report = state["report"]
        store = chunk_store()
        vector_index = index()
        by_source: dict[str, list[Document]] = defaultdict(list)
        for piece in state["chunks"]:
            by_source[str(piece.metadata.get("source", ""))].append(piece)

        pending: list[Document] = []
        stale: list[str] = []
        for source, pieces in by_source.items():
            existing = store.ids_for_source(source)
            current = {piece.metadata["chunk_id"] for piece in pieces}
            pending.extend(p for p in pieces if p.metadata["chunk_id"] not in existing)
            stale.extend(existing - current)
            report.chunks_unchanged += len(existing & current)

        # The vector store is written first and each batch is committed afterwards, so a failed
        # run can be repeated and resumes from the last completed batch.
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start : start + config.batch_size]
            vector_index.add(batch)
            store.upsert(batch)
            report.chunks_added += len(batch)
        for start in range(0, len(stale), config.batch_size):
            batch_ids = stale[start : start + config.batch_size]
            vector_index.delete(batch_ids)
            store.delete(batch_ids)
            report.chunks_removed += len(batch_ids)

        report.chunks_total = len(state["chunks"])
        return {"report": report}

    def has_documents(state: IngestState) -> str:
        return "chunk" if state["documents"] else END

    graph = StateGraph(IngestState)
    graph.add_node("load", load)
    graph.add_node("chunk", chunk)
    graph.add_node("synchronise", synchronise)
    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", has_documents, ["chunk", END])
    graph.add_edge("chunk", "synchronise")
    graph.add_edge("synchronise", END)
    return graph.compile()
