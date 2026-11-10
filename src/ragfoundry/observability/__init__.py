"""Tracing through LangSmith or Langfuse."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .._utils import require
from ..config import ObservabilityConfig
from ..errors import ConfigurationError

__all__ = ["Observer", "build_observer"]


class Observer:
    """The callbacks for a tracing backend, and a way to flush them."""

    def __init__(self, callbacks: list[BaseCallbackHandler] | None = None, client: Any = None):
        self.callbacks = callbacks or []
        self._client = client

    def flush(self) -> None:
        for callback in self.callbacks:
            wait = getattr(callback, "wait_for_futures", None)
            if wait:
                wait()
        if self._client is not None:
            self._client.flush()


def build_observer(config: ObservabilityConfig) -> Observer:
    if config.provider == "none":
        return Observer()
    if config.provider == "langsmith":
        return _langsmith(config)
    return _langfuse(config)


def _langsmith(config: ObservabilityConfig) -> Observer:
    langsmith = require("langsmith", "langsmith")
    from langchain_core.tracers.langchain import LangChainTracer

    if config.api_key is None:
        raise ConfigurationError("observability.api_key is required for LangSmith")
    client = langsmith.Client(api_url=config.endpoint, api_key=config.api_key.get_secret_value())
    tracer = LangChainTracer(client=client, project_name=config.project, tags=config.tags or None)
    return Observer([tracer], client)


def _langfuse(config: ObservabilityConfig) -> Observer:
    langfuse = require("langfuse", "langfuse")
    handler_module = require("langfuse.langchain", "langfuse")

    if config.public_key is None or config.secret_key is None:
        raise ConfigurationError(
            "observability.public_key and observability.secret_key are required for Langfuse"
        )
    client = langfuse.Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key.get_secret_value(),
        base_url=config.endpoint,
    )
    handler = handler_module.CallbackHandler(public_key=config.public_key)
    return Observer([handler], client)
