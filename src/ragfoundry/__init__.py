"""ragfoundry: configurable production RAG on LangChain and LangGraph."""

from __future__ import annotations

from .chunking import chunk_documents, register_chunker
from .config import RAGConfig
from .errors import (
    ConfigurationError,
    IngestionError,
    MissingDependencyError,
    RAGFoundryError,
)
from .evaluation import EvalReport, EvalSample, load_dataset
from .graphs import IngestReport
from .loaders import load_sources, register_loader
from .pipeline import RAGPipeline, RAGResult

__version__ = "0.1.0"

__all__ = [
    "ConfigurationError",
    "EvalReport",
    "EvalSample",
    "IngestReport",
    "IngestionError",
    "MissingDependencyError",
    "RAGConfig",
    "RAGFoundryError",
    "RAGPipeline",
    "RAGResult",
    "chunk_documents",
    "load_dataset",
    "load_sources",
    "register_chunker",
    "register_loader",
]
