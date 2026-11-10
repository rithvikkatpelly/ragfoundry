"""Loader registry and source resolution."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .._utils import require
from ..config import LoadingConfig
from ..errors import IngestionError
from .excel import ExcelLoader
from .web import WebLoader

logger = logging.getLogger(__name__)

LoaderFactory = Callable[[Path, LoadingConfig], BaseLoader]
_LOADERS: dict[str, LoaderFactory] = {}


def register_loader(*extensions: str) -> Callable[[LoaderFactory], LoaderFactory]:
    """Register a factory for file extensions such as ``".pptx"``."""

    def decorator(factory: LoaderFactory) -> LoaderFactory:
        for extension in extensions:
            _LOADERS[extension.lower()] = factory
        return factory

    return decorator


def supported_extensions() -> list[str]:
    return sorted(_LOADERS)


@register_loader(".pdf")
def _pdf(path: Path, config: LoadingConfig) -> BaseLoader:
    require("pypdf", "pdf")
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path))


@register_loader(".docx")
def _docx(path: Path, config: LoadingConfig) -> BaseLoader:
    require("docx2txt", "docx")
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path))


@register_loader(".txt", ".md", ".markdown")
def _text(path: Path, config: LoadingConfig) -> BaseLoader:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding=config.encoding)


@register_loader(".csv")
def _csv(path: Path, config: LoadingConfig) -> BaseLoader:
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(str(path), encoding=config.encoding)


@register_loader(".xlsx", ".xlsm", ".xls")
def _excel(path: Path, config: LoadingConfig) -> BaseLoader:
    return ExcelLoader(path)


@dataclass
class LoadFailure:
    source: str
    error: str


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    failures: list[LoadFailure] = field(default_factory=list)


def load_sources(sources: str | Path | Iterable[str | Path], config: LoadingConfig) -> LoadResult:
    """Load every source; failures raise or are collected according to ``config.on_error``."""
    if isinstance(sources, (str, Path)):
        sources = [sources]
    result = LoadResult()
    for source in sources:
        for target, make_loader in _expand(str(source), config, result):
            try:
                documents = _load_one(target, make_loader)
            except Exception as exc:
                _record(result, target, exc, config)
                continue
            result.documents.extend(documents)
    return result


def _expand(
    source: str, config: LoadingConfig, result: LoadResult
) -> list[tuple[str, Callable[[], BaseLoader]]]:
    if source.startswith(("http://", "https://")):
        return [(source, lambda: WebLoader(source, config.web))]
    path = Path(source).expanduser().resolve()
    try:
        if path.is_dir():
            files = sorted(p for p in path.glob(config.glob) if p.is_file())
            supported = [p for p in files if p.suffix.lower() in _LOADERS]
            logger.info("%s: %d supported of %d files", path, len(supported), len(files))
            if not supported:
                raise IngestionError(f"No supported files found in directory {path}")
        elif path.is_file():
            if path.suffix.lower() not in _LOADERS:
                raise IngestionError(
                    f"Unsupported file type '{path.suffix}'. "
                    f"Supported: {', '.join(supported_extensions())}"
                )
            supported = [path]
        else:
            raise IngestionError(f"Source not found: {source}")
    except IngestionError as exc:
        _record(result, source, exc, config)
        return []
    return [(str(p), _bind(p, config)) for p in supported]


def _bind(path: Path, config: LoadingConfig) -> Callable[[], BaseLoader]:
    return lambda: _LOADERS[path.suffix.lower()](path, config)


def _load_one(target: str, make_loader: Callable[[], BaseLoader]) -> list[Document]:
    documents = []
    for document in make_loader().load():
        if not document.page_content.strip():
            continue  # e.g. blank pages inside an otherwise readable PDF
        document.metadata.setdefault("source", target)
        document.metadata["file_type"] = _file_type(target)
        documents.append(document)
    if not documents:
        raise IngestionError(
            f"No extractable text in {target} (a scanned PDF needs OCR before ingestion)"
        )
    return documents


def _file_type(target: str) -> str:
    if target.startswith(("http://", "https://")):
        return "web"
    return Path(target).suffix.lower().lstrip(".")


def _record(result: LoadResult, target: str, exc: Exception, config: LoadingConfig) -> None:
    if config.on_error == "raise":
        if isinstance(exc, IngestionError):
            raise exc
        raise IngestionError(f"Failed to load {target}: {exc}") from exc
    logger.warning("Skipping %s: %s", target, exc)
    result.failures.append(LoadFailure(source=target, error=str(exc)))
