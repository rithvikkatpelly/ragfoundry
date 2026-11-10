"""The main entry point: ingest documents, answer questions, evaluate quality."""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ._utils import message_text
from .chunk_store import ChunkStore
from .chunking import ChunkerResources, build_splitter
from .config import RAGConfig
from .errors import ConfigurationError, IngestionError
from .evaluation.dataset import EvalSample, load_dataset
from .evaluation.runner import EvalReport, Mode, evaluate_pipeline
from .graphs import IngestReport, QueryComponents, build_ingest_graph, build_query_graph
from .models import build_chat_model, build_embeddings
from .observability import Observer, build_observer
from .retrieval import BM25Index, Reranker, Retriever, build_reranker
from .vectorstores import VectorIndex, build_vector_index


@dataclass
class RAGResult:
    question: str
    answer: str
    documents: list[Document] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    retries: int = 0
    abstained: bool = False

    @property
    def sources(self) -> list[str]:
        """Distinct document sources, in rank order."""
        names = (str(doc.metadata.get("source", "")) for doc in self.documents)
        return list(dict.fromkeys(name for name in names if name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "queries": self.queries,
            "retries": self.retries,
            "abstained": self.abstained,
            "documents": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in self.documents
            ],
        }


class RAGPipeline:
    """A configured RAG system.

    Components are created on first use, so ingest-only processes never build an LLM. Any
    LangChain-compatible ``llm``, ``embeddings`` or ``vector_store`` passed here replaces the
    configured one.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        _warn_openmp_conflict(self.config)

    @classmethod
    def from_yaml(cls, path: str | Path, **components: Any) -> RAGPipeline:
        return cls(RAGConfig.from_yaml(path), **components)

    # ------------------------------------------------------------------ components
    @cached_property
    def llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        if self.config.llm is None:
            raise ConfigurationError("No LLM configured: set 'llm' or pass llm=...")
        return build_chat_model(self.config.llm)

    @cached_property
    def judge_llm(self) -> BaseChatModel:
        judge = self.config.evaluation.llm
        return build_chat_model(judge) if judge else self.llm

    @cached_property
    def embeddings(self) -> Embeddings:
        if self._embeddings is not None:
            return self._embeddings
        if self.config.embeddings is None:
            raise ConfigurationError("No embeddings configured: set 'embeddings' or pass one")
        return build_embeddings(self.config.embeddings)

    @cached_property
    def index(self) -> VectorIndex:
        if self._vector_store is not None:
            return VectorIndex(self._vector_store)
        if self.config.vector_store is None:
            raise ConfigurationError("No vector store configured: set 'vector_store' or pass one")
        return build_vector_index(
            self.config.vector_store, self.embeddings, self.config.persist_dir
        )

    @cached_property
    def chunk_store(self) -> ChunkStore:
        store = ChunkStore(self.config.persist_dir, self._collection)
        store.ensure_fingerprint(self._embedding_fingerprint())
        return store

    @property
    def _collection(self) -> str:
        return self.config.vector_store.collection_name if self.config.vector_store else "default"

    def _embedding_fingerprint(self) -> str:
        if self.config.embeddings and self._embeddings is None:
            return f"{self.config.embeddings.provider}:{self.config.embeddings.model}"
        if self._embeddings is not None:
            return type(self._embeddings).__name__
        return "external"

    @cached_property
    def reranker(self) -> Reranker | None:
        if self._reranker is not None:
            return self._reranker
        settings = self.config.retrieval.rerank
        return build_reranker(settings) if settings else None

    @cached_property
    def observer(self) -> Observer:
        return build_observer(self.config.observability)

    @cached_property
    def retriever(self) -> Retriever:
        settings = self.config.retrieval
        bm25 = BM25Index(self.chunk_store) if settings.hybrid.enabled else None
        return Retriever(self.index, settings, bm25)

    def _components(self) -> QueryComponents:
        settings = self.config.retrieval
        needs_llm = bool(settings.query_transform.strategies) or settings.corrective.enabled
        return QueryComponents(
            config=self.config,
            retriever=self.retriever,
            llm=self.llm if needs_llm else None,
            reranker=self.reranker,
        )

    @cached_property
    def _ingest_graph(self) -> Any:
        return build_ingest_graph(
            self.config, self._make_splitter, lambda: self.index, lambda: self.chunk_store
        )

    @cached_property
    def _answer_graph(self) -> Any:
        components = self._components()
        components.llm = self.llm
        return build_query_graph(components)

    @cached_property
    def _retrieval_graph(self) -> Any:
        return build_query_graph(self._components(), generate_answer=False)

    def _make_splitter(self) -> Any:
        resources = ChunkerResources(embeddings=lambda: self.embeddings, llm=lambda: self.llm)
        return build_splitter(self.config.chunking, resources)

    def _run_config(self, name: str) -> dict[str, Any]:
        settings = self.config.observability
        return {
            "run_name": f"ragfoundry.{name}",
            "callbacks": self.observer.callbacks,
            "tags": settings.tags,
        }

    # ------------------------------------------------------------------- ingestion
    def ingest(self, sources: str | Path | Iterable[str | Path]) -> IngestReport:
        """Load, chunk and index sources; repeat runs only process what changed."""
        items = [sources] if isinstance(sources, (str, Path)) else list(sources)
        if not items:
            raise IngestionError("No sources were provided")
        state = self._ingest_graph.invoke(
            {"sources": [str(item) for item in items]}, config=self._run_config("ingest")
        )
        return state["report"]

    def sources(self) -> dict[str, int]:
        """Return indexed sources and their chunk counts."""
        return self.chunk_store.sources()

    def delete_source(self, source: str) -> int:
        """Remove a source from the index and return the number of chunks deleted."""
        ids = sorted(self.chunk_store.ids_for_source(source))
        for start in range(0, len(ids), self.config.batch_size):
            batch = ids[start : start + self.config.batch_size]
            self.index.delete(batch)
            self.chunk_store.delete(batch)
        return len(ids)

    # --------------------------------------------------------------------- queries
    @staticmethod
    def _input(question: str, where: dict[str, Any] | None, top_k: int | None) -> dict[str, Any]:
        return {"question": question, "where": where, "top_k": top_k}

    @staticmethod
    def _result(question: str, state: dict[str, Any]) -> RAGResult:
        return RAGResult(
            question=question,
            answer=state.get("answer", ""),
            documents=state.get("documents", []),
            queries=state.get("all_queries", []),
            retries=state.get("retries", 0),
            abstained=state.get("abstained", False),
        )

    def retrieve(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> list[Document]:
        """Run the retrieval stages without generating an answer."""
        state = self._retrieval_graph.invoke(
            self._input(question, where, top_k), config=self._run_config("retrieve")
        )
        limit = top_k or self.config.retrieval.top_k
        return state["documents"][:limit]

    def query(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> RAGResult:
        """Answer a question. ``where`` restricts results by exact metadata values."""
        state = self._answer_graph.invoke(
            self._input(question, where, top_k), config=self._run_config("query")
        )
        return self._result(question, state)

    async def aquery(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> RAGResult:
        """Async form of :meth:`query`."""
        state = await self._answer_graph.ainvoke(
            self._input(question, where, top_k), config=self._run_config("query")
        )
        return self._result(question, state)

    def stream(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> Iterator[str]:
        """Yield answer text as it is generated."""
        events = self._answer_graph.stream(
            self._input(question, where, top_k),
            config=self._run_config("stream"),
            stream_mode=["messages", "updates"],
        )
        for mode, payload in events:
            text = _stream_text(mode, payload)
            if text:
                yield text

    async def astream(
        self, question: str, *, where: dict[str, Any] | None = None, top_k: int | None = None
    ) -> AsyncIterator[str]:
        """Async form of :meth:`stream`."""
        events = self._answer_graph.astream(
            self._input(question, where, top_k),
            config=self._run_config("stream"),
            stream_mode=["messages", "updates"],
        )
        async for mode, payload in events:
            text = _stream_text(mode, payload)
            if text:
                yield text

    # ------------------------------------------------------------------ evaluation
    def evaluate(
        self,
        dataset: str | Path | Sequence[EvalSample],
        *,
        mode: Mode = "all",
        backend: Literal["native", "ragas"] = "native",
        k_values: Sequence[int] | None = None,
    ) -> EvalReport:
        """Score the pipeline against a dataset file or a list of samples."""
        samples = load_dataset(dataset) if isinstance(dataset, (str, Path)) else list(dataset)
        return evaluate_pipeline(self, samples, mode=mode, backend=backend, k_values=k_values)

    # ------------------------------------------------------------------- lifecycle
    def flush(self) -> None:
        """Send any buffered traces."""
        if "observer" in self.__dict__:
            self.observer.flush()

    def close(self) -> None:
        """Flush traces and release vector store connections."""
        self.flush()
        if "index" in self.__dict__:
            self.index.close()

    def __enter__(self) -> RAGPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _warn_openmp_conflict(config: RAGConfig) -> None:
    """Warn about a FAISS and PyTorch combination that can abort the process on macOS."""
    store = config.vector_store
    uses_torch = (
        config.embeddings is not None and config.embeddings.provider == "huggingface"
    ) or (config.retrieval.rerank is not None and config.retrieval.rerank.provider == "huggingface")
    if sys.platform == "darwin" and store is not None and store.provider == "faiss" and uses_torch:
        message = (
            "On macOS the faiss-cpu and torch wheels each bundle their own OpenMP runtime, and "
            "using both in one process can abort it ('OMP: Error #15'). Prefer Chroma or Qdrant "
            "with Hugging Face models on macOS."
        )
        logging.getLogger(__name__).warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _stream_text(mode: str, payload: Any) -> str:
    if mode == "messages":
        chunk, metadata = payload
        if metadata.get("langgraph_node") == "generate":
            return _chunk_text(chunk)
        return ""
    if isinstance(payload, dict) and "abstain" in payload:
        return payload["abstain"]["answer"]
    return ""


def _chunk_text(chunk: Any) -> str:
    # Whitespace is meaningful between tokens, so message_text's stripping cannot be used here.
    if isinstance(chunk.content, str):
        return chunk.content
    return message_text(chunk)
