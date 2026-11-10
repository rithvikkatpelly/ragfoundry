"""Build each supported vector store from its typed configuration."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings

from .._utils import require
from ..config import (
    ChromaConfig,
    FAISSConfig,
    MilvusConfig,
    PGVectorConfig,
    PineconeConfig,
    QdrantConfig,
    WeaviateConfig,
)
from .index import VectorIndex


def build_vector_index(config: Any, embeddings: Embeddings, persist_dir: str | Path) -> VectorIndex:
    builders: dict[str, Callable[[Any, Embeddings, Path], VectorIndex]] = {
        "chroma": _chroma,
        "faiss": _faiss,
        "pinecone": _pinecone,
        "qdrant": _qdrant,
        "weaviate": _weaviate,
        "milvus": _milvus,
        "pgvector": _pgvector,
    }
    return builders[config.provider](config, embeddings, Path(persist_dir))


def _chroma(config: ChromaConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_chroma", "chroma")
    location: dict[str, Any]
    if config.host:
        location = {"host": config.host, "port": config.port or 8000}
    else:
        location = {"persist_directory": config.persist_directory or str(persist_dir / "chroma")}
    store = module.Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        **location,
    )

    def translate(where: dict[str, Any]) -> dict[str, Any]:
        if len(where) == 1:
            return {"filter": where}
        return {"filter": {"$and": [{key: value} for key, value in where.items()]}}

    return VectorIndex(store, translate_filter=translate)


def _faiss(config: FAISSConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    faiss = require("faiss", "faiss")
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    # Inner product over normalised vectors gives cosine ranking, matching the other stores.
    settings: dict[str, Any] = {
        "normalize_L2": True,
        "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
    }
    path = Path(config.path) if config.path else persist_dir / "faiss" / config.collection_name
    # LangChain warns that normalize_L2 "is not applicable" to inner product, yet it still
    # normalises (which is what makes this cosine), so the misleading warning is suppressed.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable")
        store = _open_faiss(faiss, FAISS, InMemoryDocstore, path, embeddings, settings)

    def remove_existing(ids: Sequence[str]) -> None:
        # FAISS rejects duplicate ids, which would otherwise break resuming a failed ingest.
        present = set(store.index_to_docstore_id.values())
        duplicates = [item for item in ids if item in present]
        if duplicates:
            store.delete(duplicates)

    return VectorIndex(
        store,
        save=lambda: store.save_local(str(path)),
        before_add=remove_existing,
    )


def _open_faiss(
    faiss: Any,
    store_class: Any,
    docstore_class: Any,
    path: Path,
    embeddings: Embeddings,
    settings: dict[str, Any],
) -> Any:
    if (path / "index.faiss").exists():
        # FAISS metadata is a pickle. This path is the store this package wrote for the user.
        return store_class.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True, **settings
        )
    dimension = len(embeddings.embed_query("dimension probe"))
    return store_class(
        embedding_function=embeddings,
        index=faiss.IndexFlatIP(dimension),
        docstore=docstore_class(),
        index_to_docstore_id={},
        **settings,
    )


def _pinecone(config: PineconeConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_pinecone", "pinecone")
    credentials = {"pinecone_api_key": config.api_key.get_secret_value()} if config.api_key else {}
    store = module.PineconeVectorStore(
        index_name=config.index_name,
        embedding=embeddings,
        namespace=config.collection_name,
        **credentials,
    )
    return VectorIndex(store)


def _qdrant(config: QdrantConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_qdrant", "qdrant")
    client_module = require("qdrant_client", "qdrant")
    models = client_module.models
    if config.url:
        api_key = config.api_key.get_secret_value() if config.api_key else None
        client = client_module.QdrantClient(url=config.url, api_key=api_key)
    else:
        client = client_module.QdrantClient(path=config.path or str(persist_dir / "qdrant"))
    if not client.collection_exists(config.collection_name):
        dimension = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            config.collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
    store = module.QdrantVectorStore(
        client=client, collection_name=config.collection_name, embedding=embeddings
    )

    def translate(where: dict[str, Any]) -> dict[str, Any]:
        conditions = [
            models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
            for key, value in where.items()
        ]
        return {"filter": models.Filter(must=conditions)}

    return VectorIndex(store, translate_filter=translate, close=client.close)


def _weaviate(config: WeaviateConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_weaviate", "weaviate")
    weaviate = require("weaviate", "weaviate")
    parts = urlparse(config.url)
    if config.api_key:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=config.url,
            auth_credentials=weaviate.auth.AuthApiKey(config.api_key.get_secret_value()),
        )
    else:
        client = weaviate.connect_to_local(
            host=parts.hostname or "localhost",
            port=parts.port or 8080,
            grpc_port=config.grpc_port,
        )
    # Weaviate class names must begin with a capital and contain no hyphens.
    name = config.collection_name.replace("-", "_")
    store = module.WeaviateVectorStore(
        client=client,
        index_name=name[0].upper() + name[1:],
        text_key="text",
        embedding=embeddings,
    )

    def translate(where: dict[str, Any]) -> dict[str, Any]:
        query = require("weaviate.classes.query", "weaviate")
        conditions = [query.Filter.by_property(k).equal(v) for k, v in where.items()]
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = combined & condition
        return {"filters": combined}

    return VectorIndex(store, translate_filter=translate, close=client.close)


def _milvus(config: MilvusConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_milvus", "milvus")
    uri = config.uri
    if uri is None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        uri = str((persist_dir / "milvus.db").resolve())
    connection: dict[str, Any] = {"uri": uri}
    if config.token:
        connection["token"] = config.token.get_secret_value()
    store = module.Milvus(
        embedding_function=embeddings,
        collection_name=config.collection_name.replace("-", "_"),
        connection_args=connection,
        auto_id=False,
        enable_dynamic_field=True,
    )

    def translate(where: dict[str, Any]) -> dict[str, Any]:
        expressions = []
        for key, value in where.items():
            if isinstance(value, bool):
                literal = "true" if value else "false"
            elif isinstance(value, str):
                literal = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
            else:
                literal = str(value)
            expressions.append(f"{key} == {literal}")
        return {"expr": " and ".join(expressions)}

    return VectorIndex(store, translate_filter=translate)


def _pgvector(config: PGVectorConfig, embeddings: Embeddings, persist_dir: Path) -> VectorIndex:
    module = require("langchain_postgres", "pgvector")
    store = module.PGVector(
        embeddings=embeddings,
        connection=config.connection.get_secret_value(),
        collection_name=config.collection_name,
        use_jsonb=True,
    )
    return VectorIndex(store)
