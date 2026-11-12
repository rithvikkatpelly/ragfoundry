"""ragfoundry: configurable production RAG on LangChain and LangGraph."""

from __future__ import annotations

from .chat import ChatSession
from .chunking import chunk_documents, register_chunker
from .config import RAGConfig
from .context import pack_documents, window_history
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

__version__ = "0.2.0"

__all__ = [
    "ChatSession",
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
    "pack_documents",
    "register_chunker",
    "register_loader",
    "window_history",
]
