"""Spreadsheet loader producing one document per data row."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .._utils import require


class ExcelLoader(BaseLoader):
    """Read every sheet of an .xlsx, .xlsm or .xls file.

    The first non-empty row of a sheet is treated as its header. Each later row becomes a
    ``header: value`` document, which keeps column meaning attached to the value when embedded.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def lazy_load(self) -> Iterator[Document]:
        for sheet, rows in self._sheets():
            yield from self._documents(sheet, rows)

    def _sheets(self) -> Iterator[tuple[str, Iterable[tuple[Any, ...]]]]:
        if self.path.suffix.lower() == ".xls":
            xlrd = require("xlrd", "excel")
            book = xlrd.open_workbook(str(self.path))
            for sheet in book.sheets():
                yield sheet.name, (tuple(sheet.row_values(i)) for i in range(sheet.nrows))
            return
        openpyxl = require("openpyxl", "excel")
        book = openpyxl.load_workbook(self.path, read_only=True, data_only=True)
        try:
            for sheet in book.worksheets:
                yield sheet.title, sheet.iter_rows(values_only=True)
        finally:
            book.close()

    def _documents(self, sheet: str, rows: Iterable[tuple[Any, ...]]) -> Iterator[Document]:
        header: list[str] | None = None
        for number, row in enumerate(rows, start=1):
            cells = ["" if cell is None else _cell_text(cell) for cell in row]
            if not any(cells):
                continue
            if header is None:
                header = [cell or f"column_{i}" for i, cell in enumerate(cells, start=1)]
                continue
            lines = [f"{name}: {cell}" for name, cell in zip(header, cells, strict=False) if cell]
            yield Document(
                page_content="\n".join(lines),
                metadata={"source": str(self.path), "sheet": sheet, "row": number},
            )


def _cell_text(cell: Any) -> str:
    # xlrd reports every number as a float, so whole numbers would otherwise read "2024.0".
    if isinstance(cell, float) and cell.is_integer():
        return str(int(cell))
    return str(cell).strip()
