"""Typed configuration for a RAG pipeline, loadable from Python or YAML."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .errors import ConfigurationError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- loading
class WebConfig(_Model):
    max_depth: int = Field(0, ge=0, description="Link depth to follow; 0 fetches only the URL.")
    max_pages: int = Field(50, ge=1)
    same_domain: bool = True
    respect_robots: bool = True
    delay_seconds: float = Field(0.5, ge=0)
    timeout_seconds: float = Field(30, gt=0)
    max_bytes: int = Field(10_000_000, gt=0)
    user_agent: str = "ragfoundry/0.2"


class LoadingConfig(_Model):
    on_error: Literal["raise", "skip"] = "raise"
    encoding: str = "utf-8"
    glob: str = "**/*"
    web: WebConfig = Field(default_factory=WebConfig)


# -------------------------------------------------------------------------- chunking
class SemanticChunkingConfig(_Model):
    breakpoint_percentile: float = Field(95, gt=0, le=100)
    buffer_size: int = Field(1, ge=0, description="Neighbouring sentences embedded with each one.")


class LLMChunkingConfig(_Model):
    window_chars: int = Field(8000, gt=0, description="Text sent to the LLM per request.")


class ChunkingConfig(_Model):
    strategy: str = Field("recursive", description="fixed, recursive, semantic, llm, or custom.")
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(200, ge=0)
    length_unit: Literal["characters", "tokens"] = "characters"
    encoding_name: str = "cl100k_base"
    separators: list[str] | None = None
    semantic: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)
    llm: LLMChunkingConfig = Field(default_factory=LLMChunkingConfig)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


# ------------------------------------------------------------------- models
class EmbeddingConfig(_Model):
    provider: Literal["huggingface", "openai", "gemini"]
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(_Model):
    provider: Literal["groq", "openai", "gemini", "claude", "ollama"]
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float | None = Field(None, description="Omitted from requests when unset.")
    max_tokens: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------- vector stores
class _VectorStoreBase(_Model):
    collection_name: str = "ragfoundry"


class ChromaConfig(_VectorStoreBase):
    provider: Literal["chroma"] = "chroma"
    persist_directory: str | None = None
    host: str | None = None
    port: int | None = None


class FAISSConfig(_VectorStoreBase):
    provider: Literal["faiss"] = "faiss"
    path: str | None = None


class PineconeConfig(_VectorStoreBase):
    provider: Literal["pinecone"] = "pinecone"
    index_name: str
    api_key: SecretStr | None = None


class QdrantConfig(_VectorStoreBase):
    provider: Literal["qdrant"] = "qdrant"
    url: str | None = None
    path: str | None = None
    api_key: SecretStr | None = None


class WeaviateConfig(_VectorStoreBase):
    provider: Literal["weaviate"] = "weaviate"
    url: str = "http://localhost:8080"
    grpc_port: int = 50051
    api_key: SecretStr | None = None


class MilvusConfig(_VectorStoreBase):
    provider: Literal["milvus"] = "milvus"
    uri: str | None = None
    token: SecretStr | None = None


class PGVectorConfig(_VectorStoreBase):
    provider: Literal["pgvector"] = "pgvector"
    connection: SecretStr


VectorStoreConfig = Annotated[
    ChromaConfig
    | FAISSConfig
    | PineconeConfig
    | QdrantConfig
    | WeaviateConfig
    | MilvusConfig
    | PGVectorConfig,
    Field(discriminator="provider"),
]


# ------------------------------------------------------------------------ retrieval
class HybridConfig(_Model):
    enabled: bool = False
    rrf_k: int = Field(60, gt=0)
    vector_weight: float = Field(1.0, ge=0)
    bm25_weight: float = Field(1.0, ge=0)


class RerankConfig(_Model):
    provider: Literal["huggingface", "cohere"]
    model: str
    api_key: SecretStr | None = None


class QueryTransformConfig(_Model):
    strategies: list[Literal["rewrite", "multi_query", "hyde"]] = Field(default_factory=list)
    num_queries: int = Field(3, ge=1, description="Alternative queries for multi_query.")


class CorrectiveConfig(_Model):
    enabled: bool = False
    max_retries: int = Field(1, ge=0)
    min_relevant_docs: int = Field(1, ge=1)


class RetrievalConfig(_Model):
    top_k: int = Field(5, gt=0, description="Documents returned to the generator.")
    fetch_k: int = Field(20, gt=0, description="Candidates per query before fusion and reranking.")
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    rerank: RerankConfig | None = None
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    corrective: CorrectiveConfig = Field(default_factory=CorrectiveConfig)

    @model_validator(mode="after")
    def _fetch_covers_top(self) -> RetrievalConfig:
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be greater than or equal to top_k")
        return self


class ContextWindowConfig(_Model):
    """Limits on what goes into the generation prompt."""

    max_context_tokens: int | None = Field(
        None, gt=0, description="Token budget for retrieved passages; unset sends all top_k."
    )
    truncate_last_passage: bool = Field(
        True, description="Shorten the passage that crosses the budget instead of dropping it."
    )
    max_history_turns: int = Field(
        5, ge=0, description="Recent user questions, with their answers, sent to the LLM."
    )
    max_history_tokens: int | None = Field(
        None, gt=0, description="Token budget for chat history; older turns are dropped first."
    )
    condense_question: bool = Field(
        True, description="Rewrite follow-up questions into standalone ones before retrieval."
    )
    token_counter: Literal["approximate", "tiktoken"] = Field(
        "approximate", description="approximate is 4 characters per token; tiktoken is exact."
    )
    encoding_name: str = Field("cl100k_base", description="tiktoken encoding to count with.")


class GenerationConfig(_Model):
    system_prompt: str | None = Field(None, description="Replaces the default system prompt.")
    abstain_message: str = "I don't have enough information in the provided documents to answer."


# --------------------------------------------------------------------- observability
class ObservabilityConfig(_Model):
    provider: Literal["none", "langsmith", "langfuse"] = "none"
    project: str = "ragfoundry"
    api_key: SecretStr | None = Field(None, description="LangSmith API key.")
    public_key: str | None = Field(None, description="Langfuse public key.")
    secret_key: SecretStr | None = Field(None, description="Langfuse secret key.")
    endpoint: str | None = Field(None, description="Self-hosted or regional server URL.")
    tags: list[str] = Field(default_factory=list)


class EvaluationConfig(_Model):
    llm: LLMConfig | None = Field(None, description="Judge model; defaults to the pipeline LLM.")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    max_concurrency: int = Field(1, ge=1)


# ------------------------------------------------------------------------------ root
class RAGConfig(_Model):
    persist_dir: str = ".ragfoundry"
    batch_size: int = Field(64, gt=0, description="Chunks written to the vector store per batch.")
    loading: LoadingConfig = Field(default_factory=LoadingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig | None = None
    vector_store: VectorStoreConfig | None = None
    llm: LLMConfig | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context_window: ContextWindowConfig = Field(default_factory=ContextWindowConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Read a YAML file, expanding ``${VAR}`` and ``${VAR:-default}`` in string values."""
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Could not read config '{path}': {exc}") from exc
        return cls.model_validate(_expand_env(raw or {}))


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(_env_value, value)
    return value


def _env_value(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    resolved = os.environ.get(name, default)
    if resolved is None:
        raise ConfigurationError(f"Environment variable '{name}' is referenced but not set")
    return resolved
