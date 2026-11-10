"""Factories for embedding models and chat models."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .._utils import require
from ..config import EmbeddingConfig, LLMConfig

__all__ = ["build_chat_model", "build_embeddings"]


def _key(config: EmbeddingConfig | LLMConfig) -> dict[str, Any]:
    """Pass a key only when configured, so each SDK can fall back to its own env variable."""
    return {"api_key": config.api_key.get_secret_value()} if config.api_key else {}


def build_embeddings(config: EmbeddingConfig) -> Embeddings:
    options = dict(config.options)
    if config.provider == "huggingface":
        module = require("langchain_huggingface", "huggingface")
        return module.HuggingFaceEmbeddings(model_name=config.model, **options)
    if config.provider == "openai":
        module = require("langchain_openai", "openai")
        if config.base_url:
            options["base_url"] = config.base_url
        return module.OpenAIEmbeddings(model=config.model, **_key(config), **options)
    module = require("langchain_google_genai", "gemini")
    return module.GoogleGenerativeAIEmbeddings(model=config.model, **_key(config), **options)


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    options = dict(config.options)
    if config.temperature is not None:
        options["temperature"] = config.temperature

    if config.provider == "ollama":
        module = require("langchain_ollama", "ollama")
        if config.base_url:
            options["base_url"] = config.base_url
        if config.max_tokens is not None:
            options["num_predict"] = config.max_tokens
        return module.ChatOllama(model=config.model, **options)

    if config.max_tokens is not None:
        options["max_tokens"] = config.max_tokens
    if config.provider == "groq":
        module = require("langchain_groq", "groq")
        return module.ChatGroq(model=config.model, **_key(config), **options)
    if config.provider == "openai":
        module = require("langchain_openai", "openai")
        if config.base_url:
            options["base_url"] = config.base_url
        return module.ChatOpenAI(model=config.model, **_key(config), **options)
    if config.provider == "gemini":
        module = require("langchain_google_genai", "gemini")
        return module.ChatGoogleGenerativeAI(model=config.model, **_key(config), **options)
    module = require("langchain_anthropic", "claude")
    if config.base_url:
        options["base_url"] = config.base_url
    return module.ChatAnthropic(model=config.model, **_key(config), **options)
