"""Exceptions raised by ragfoundry."""

from __future__ import annotations


class RAGFoundryError(Exception):
    """Base class for all ragfoundry errors."""


class ConfigurationError(RAGFoundryError):
    """The supplied configuration is invalid or incomplete."""


class MissingDependencyError(RAGFoundryError, ImportError):
    """An optional dependency is required for the selected feature."""


class IngestionError(RAGFoundryError):
    """A document could not be loaded or indexed."""
