"""Small helpers shared across the package."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any

from langchain_core.messages import BaseMessage

from .errors import MissingDependencyError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_METADATA_KEY = re.compile(r"[^0-9a-zA-Z_]")


def require(module: str, extra: str) -> Any:
    """Import an optional module, or explain which extra provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"'{module}' is required for this feature. "
            f"Install it with: pip install 'ragfoundry[{extra}]'"
        ) from exc


def message_text(message: BaseMessage | str) -> str:
    """Return the plain text of a model reply, without reasoning blocks."""
    content: Any = message if isinstance(message, str) else message.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = "".join(parts)
    return _THINK_BLOCK.sub("", str(content)).strip()


def parse_json(text: str) -> Any | None:
    """Extract the first JSON array or object from model output, or ``None``."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        return value
    return None


def string_items(value: Any) -> list[str] | None:
    """Read a model's JSON list of strings, tolerating ``[{"claim": "..."}]`` style objects.

    Returns ``None`` when the value is not a list at all.
    """
    if not isinstance(value, list):
        return None
    items = []
    for item in value:
        if isinstance(item, dict):
            item = next((v for v in item.values() if isinstance(v, str)), None)
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Make metadata safe for every vector store.

    Stores differ in what they accept, so keys are reduced to identifier characters, ``None``
    values are dropped and non-scalar values are JSON encoded.
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        name = _METADATA_KEY.sub("_", str(key))
        if not name or name[0].isdigit():
            name = f"m_{name}"
        if isinstance(value, (str, int, float, bool)):
            clean[name] = value
        else:
            clean[name] = json.dumps(value, default=str, ensure_ascii=False)
    return clean
