"""Vector store construction and a uniform indexing interface."""

from __future__ import annotations

from .factory import build_vector_index
from .index import VectorIndex

__all__ = ["VectorIndex", "build_vector_index"]
