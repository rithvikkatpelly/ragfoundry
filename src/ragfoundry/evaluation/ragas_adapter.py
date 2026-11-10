"""Optional RAGAS backend for generation metrics."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ..errors import MissingDependencyError

_REMOVED_MODULE = "langchain_community.chat_models.vertexai"


def _import_ragas() -> Any:
    """Import RAGAS, tolerating a LangChain module it references but no longer exists.

    RAGAS 0.4 imports ``ChatVertexAI`` from that path only for an ``isinstance`` check, and current
    ``langchain-community`` has removed it. A placeholder class is safe because no real model is
    ever an instance of it, and it is only installed when the real module is absent.
    """
    try:
        importlib.import_module(_REMOVED_MODULE)
    except ImportError:
        placeholder = types.ModuleType(_REMOVED_MODULE)
        placeholder.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules.setdefault(_REMOVED_MODULE, placeholder)
    try:
        return importlib.import_module("ragas")
    except ImportError as exc:
        raise MissingDependencyError(
            "RAGAS could not be imported. Install it with: pip install 'ragfoundry[ragas]'"
        ) from exc


def evaluate_with_ragas(
    rows: Sequence[dict[str, Any]],
    llm: BaseChatModel,
    embeddings: Embeddings,
    metric_names: Sequence[str] | None = None,
) -> list[dict[str, float | None]]:
    """Score rows of ``question``, ``answer``, ``contexts`` and optional ``reference``.

    Returns one dictionary of metric values per row. Rows without contexts cannot be scored by
    RAGAS, so they are skipped and reported as an empty dictionary in their original position.
    """
    ragas = _import_ragas()
    metrics_module = importlib.import_module("ragas.metrics")

    scorable = [index for index, row in enumerate(rows) if row.get("contexts")]
    results: list[dict[str, float | None]] = [{} for _ in rows]
    if not scorable:
        return results
    has_reference = all(rows[index].get("reference") for index in scorable)
    names = list(metric_names or ["faithfulness", "answer_relevancy"])
    if metric_names is None and has_reference:
        names += ["context_precision", "context_recall"]
    metrics = [getattr(metrics_module, name) for name in names]

    samples = []
    for index in scorable:
        row = rows[index]
        sample = {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": list(row["contexts"]),
        }
        if has_reference:
            sample["reference"] = row["reference"]
        samples.append(sample)
    dataset = ragas.EvaluationDataset.from_list(samples)
    outcome = ragas.evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=embeddings, show_progress=False
    )
    for index, scores in zip(scorable, outcome.scores, strict=True):
        results[index] = {
            name: None if value is None or value != value else float(value)
            for name, value in scores.items()
        }
    return results
