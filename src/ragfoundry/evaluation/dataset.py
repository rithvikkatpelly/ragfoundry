"""Evaluation dataset schema and file loading."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ConfigurationError


class EvalSample(BaseModel):
    """One test question. Retrieval metrics need at least one of the three relevance fields."""

    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str | None = None
    relevant_ids: list[str] = Field(default_factory=list, description="Exact chunk ids.")
    relevant_sources: list[str] = Field(
        default_factory=list, description="Source URLs, paths, or path suffixes such as a filename."
    )
    reference_contexts: list[str] = Field(
        default_factory=list, description="Passages that a relevant chunk should overlap."
    )

    @property
    def has_relevance_labels(self) -> bool:
        return bool(self.relevant_ids or self.relevant_sources or self.reference_contexts)


def load_dataset(path: str | Path) -> list[EvalSample]:
    """Read samples from a ``.json`` array or ``.jsonl`` file."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
        if file.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read dataset '{path}': {exc}") from exc
    if not isinstance(records, list) or not records:
        raise ConfigurationError(f"Dataset '{path}' must contain a non-empty list of samples")
    return [EvalSample.model_validate(record) for record in records]
