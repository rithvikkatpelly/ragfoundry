"""Sentence segmentation that keeps offsets into the original text."""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n\s*\n")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of non-blank sentences."""
    spans = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans
