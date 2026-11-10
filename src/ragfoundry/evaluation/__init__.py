"""Retrieval and generation evaluation."""

from __future__ import annotations

from .dataset import EvalSample, load_dataset
from .generation import (
    answer_correctness,
    answer_relevance,
    context_recall,
    context_relevance,
    faithfulness,
)
from .ragas_adapter import evaluate_with_ragas
from .retrieval import average_precision, hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .runner import EvalReport, evaluate_generation_rows, evaluate_pipeline

__all__ = [
    "EvalReport",
    "EvalSample",
    "answer_correctness",
    "answer_relevance",
    "average_precision",
    "context_recall",
    "context_relevance",
    "evaluate_generation_rows",
    "evaluate_pipeline",
    "evaluate_with_ragas",
    "faithfulness",
    "hit_rate",
    "load_dataset",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
