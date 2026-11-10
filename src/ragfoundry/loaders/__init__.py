"""Turn files, directories and URLs into LangChain documents."""

from __future__ import annotations

from .base import LoadFailure, LoadResult, load_sources, register_loader, supported_extensions
from .web import WebLoader

__all__ = [
    "LoadFailure",
    "LoadResult",
    "WebLoader",
    "load_sources",
    "register_loader",
    "supported_extensions",
]
