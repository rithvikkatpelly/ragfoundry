"""SQLite record of indexed chunks.

The record makes re-ingestion incremental, supplies the corpus for BM25, and detects an embedding
model change that would otherwise mix incompatible vectors in one collection.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.documents import Document

from .errors import ConfigurationError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class ChunkStore:
    """Durable map of chunk ids to their content. Safe to share between threads."""

    def __init__(self, persist_dir: str | Path, collection: str) -> None:
        directory = Path(persist_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{re.sub(r'[^0-9A-Za-z_-]', '_', collection)}.chunks.sqlite"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ensure_fingerprint(self, fingerprint: str) -> None:
        """Bind the collection to one embedding model, or fail if it already has another."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (fingerprint,))
            elif row[0] != fingerprint:
                raise ConfigurationError(
                    f"This collection was indexed with embeddings '{row[0]}' but the pipeline "
                    f"is configured with '{fingerprint}'. Use a new collection_name or re-index."
                )

    def revision(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
        return int(row[0]) if row else 0

    def ids_for_source(self, source: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM chunks WHERE source=?", (source,)).fetchall()
        return {row[0] for row in rows}

    def sources(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def documents(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT content, metadata FROM chunks ORDER BY id").fetchall()
        return [Document(page_content=row[0], metadata=json.loads(row[1])) for row in rows]

    def upsert(self, chunks: Iterable[Document]) -> None:
        rows = [
            (
                chunk.metadata["chunk_id"],
                str(chunk.metadata.get("source", "")),
                chunk.page_content,
                json.dumps(chunk.metadata, default=str, ensure_ascii=False),
            )
            for chunk in chunks
        ]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)", rows)
            self._bump(conn)

    def delete(self, ids: Iterable[str]) -> None:
        with self._connect() as conn:
            conn.executemany("DELETE FROM chunks WHERE id=?", [(item,) for item in ids])
            self._bump(conn)

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO meta VALUES ('revision', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )
