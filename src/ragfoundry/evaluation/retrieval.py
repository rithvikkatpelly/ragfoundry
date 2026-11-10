"""Rank-based retrieval metrics.

``retrieved`` is the ranked list of item keys and ``relevant`` the gold set. A key may repeat in
``retrieved``, for example when several chunks come from one relevant document. Each repeat counts
towards precision, but only the first occurrence counts as finding that item in the rest.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _cut(retrieved: Sequence[str], k: int | None) -> Sequence[str]:
    return retrieved if k is None else retrieved[:k]


def _first_hits(retrieved: Sequence[str], relevant: Collection[str]) -> list[bool]:
    seen: set[str] = set()
    hits = []
    for item in retrieved:
        hits.append(item in relevant and item not in seen)
        seen.add(item)
    return hits


def hit_rate(retrieved: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    """1 if any relevant item appears in the top ``k``, otherwise 0."""
    return float(any(item in relevant for item in _cut(retrieved, k)))


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / k


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    return len(set(_cut(retrieved, k)) & set(relevant)) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant item."""
    for rank, item in enumerate(_cut(retrieved, k), start=1):
        if item in relevant:
            return 1 / rank
    return 0.0


def average_precision(
    retrieved: Sequence[str], relevant: Collection[str], k: int | None = None
) -> float:
    if not relevant:
        return 0.0
    found = 0
    total = 0.0
    for rank, hit in enumerate(_first_hits(_cut(retrieved, k), relevant), start=1):
        if hit:
            found += 1
            total += found / rank
    return total / len(relevant)


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance normalised discounted cumulative gain."""
    if not relevant:
        return 0.0
    hits = _first_hits(retrieved[:k], relevant)
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal
