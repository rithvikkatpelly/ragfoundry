"""Run a dataset through a pipeline and aggregate the results."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from . import generation as gen
from . import retrieval as ret
from .dataset import EvalSample
from .ragas_adapter import evaluate_with_ragas

if TYPE_CHECKING:
    from ..pipeline import RAGPipeline

logger = logging.getLogger(__name__)
Mode = Literal["retrieval", "generation", "all"]


@dataclass
class EvalReport:
    """Per-sample rows plus, for each metric, its mean and the number of scored samples."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "rows": self.rows}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )


def retrieval_keys(documents: Sequence[Document], sample: EvalSample) -> tuple[list[str], set[str]]:
    """Map retrieved documents onto gold keys, so all label types share one set of metrics."""
    gold: set[str] = set()
    gold.update(f"id:{item}" for item in sample.relevant_ids)
    gold.update(f"source:{item}" for item in sample.relevant_sources)
    gold.update(f"context:{index}" for index in range(len(sample.reference_contexts)))

    keys: list[str] = []
    found: set[str] = set()
    for position, document in enumerate(documents):
        matches = _matches(document, sample)
        fresh = [key for key in matches if key not in found]
        key = (fresh or matches or [f"miss:{position}"])[0]
        found.add(key)
        keys.append(key)
    return keys, gold


def _matches(document: Document, sample: EvalSample) -> list[str]:
    matches = []
    if document.metadata.get("chunk_id") in sample.relevant_ids:
        matches.append(f"id:{document.metadata['chunk_id']}")
    source = str(document.metadata.get("source", ""))
    for item in sample.relevant_sources:
        if source == item or source.endswith(("/" + item.lstrip("/"), "\\" + item.lstrip("\\"))):
            matches.append(f"source:{item}")
    text = _normalize(document.page_content)
    for index, passage in enumerate(sample.reference_contexts):
        target = _normalize(passage)
        if target and (target in text or text in target):
            matches.append(f"context:{index}")
    return matches


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score_retrieval(
    documents: Sequence[Document], sample: EvalSample, k_values: Sequence[int]
) -> dict[str, float]:
    keys, gold = retrieval_keys(documents, sample)
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"hit_rate@{k}"] = ret.hit_rate(keys, gold, k)
        scores[f"precision@{k}"] = ret.precision_at_k(keys, gold, k)
        scores[f"recall@{k}"] = ret.recall_at_k(keys, gold, k)
        scores[f"ndcg@{k}"] = ret.ndcg_at_k(keys, gold, k)
    scores["mrr"] = ret.mrr(keys, gold)
    scores["map"] = ret.average_precision(keys, gold)
    return scores


def score_generation(llm: BaseChatModel, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Compute native metrics for a row, isolating failures to the metric that raised."""
    question, answer = row["question"], row["answer"]
    contexts, reference = row["contexts"], row.get("reference")
    judges: dict[str, Callable[[], float | None]] = {
        "faithfulness": lambda: gen.faithfulness(llm, answer, contexts, question),
        "answer_relevance": lambda: gen.answer_relevance(llm, question, answer),
        "context_relevance": lambda: gen.context_relevance(llm, question, contexts),
    }
    if reference:
        judges["answer_correctness"] = lambda: gen.answer_correctness(
            llm, question, answer, reference
        )
        judges["context_recall"] = lambda: gen.context_recall(llm, reference, contexts, question)

    scores: dict[str, Any] = {}
    errors: list[str] = []
    for name, judge in judges.items():
        try:
            scores[name] = judge()
        except Exception as exc:
            logger.warning("Metric %s failed: %s", name, exc)
            scores[name] = None
            errors.append(f"{name}: {exc}")
    return scores, errors


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    collected: dict[str, list[float]] = {}
    for row in rows:
        for section in ("retrieval_metrics", "generation_metrics"):
            for name, value in row.get(section, {}).items():
                values = collected.setdefault(name, [])
                if value is not None:
                    values.append(float(value))
    return {
        name: {"mean": sum(values) / len(values) if values else float("nan"), "count": len(values)}
        for name, values in sorted(collected.items())
    }


def evaluate_generation_rows(
    rows: list[dict[str, Any]],
    llm: BaseChatModel,
    embeddings: Any = None,
    backend: Literal["native", "ragas"] = "native",
) -> None:
    """Add ``generation_metrics`` to rows holding question, answer, contexts and reference."""
    if backend == "ragas":
        if embeddings is None:
            raise ValueError("The RAGAS backend requires embeddings")
        for row, scores in zip(rows, evaluate_with_ragas(rows, llm, embeddings), strict=True):
            row["generation_metrics"] = scores
        return
    for row in rows:
        row["generation_metrics"], errors = score_generation(llm, row)
        if errors:
            row["errors"] = errors


def evaluate_pipeline(
    pipeline: RAGPipeline,
    samples: Sequence[EvalSample],
    *,
    mode: Mode = "all",
    backend: Literal["native", "ragas"] = "native",
    k_values: Sequence[int] | None = None,
) -> EvalReport:
    """Run samples through the pipeline and score retrieval, generation, or both."""
    settings = pipeline.config.evaluation
    k_values = list(k_values or settings.k_values)
    wants_retrieval = mode in ("retrieval", "all")
    wants_generation = mode in ("generation", "all")

    def run(sample: EvalSample) -> dict[str, Any]:
        row: dict[str, Any] = {"question": sample.question, "reference": sample.reference_answer}
        if wants_generation:
            result = pipeline.query(sample.question, top_k=max(k_values))
            documents, row["answer"] = result.documents, result.answer
            row["abstained"] = result.abstained
        else:
            documents = pipeline.retrieve(sample.question, top_k=max(k_values))
        row["contexts"] = [document.page_content for document in documents]
        row["retrieved_sources"] = [document.metadata.get("source") for document in documents]
        if wants_retrieval and sample.has_relevance_labels:
            row["retrieval_metrics"] = score_retrieval(documents, sample, k_values)
        return row

    with ThreadPoolExecutor(max_workers=settings.max_concurrency) as executor:
        rows = list(executor.map(run, samples))

    if wants_generation:
        evaluate_generation_rows(rows, pipeline.judge_llm, pipeline.embeddings, backend)
    return EvalReport(rows=rows, summary=summarize(rows))
