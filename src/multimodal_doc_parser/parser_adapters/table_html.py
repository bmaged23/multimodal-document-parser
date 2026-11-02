"""Convert parser-produced HTML tables into validated cell grids."""

from html.parser import HTMLParser
from typing import Any

from multimodal_doc_parser.document_schema.document import TableCell


class TableHTMLParser(HTMLParser):
    """Read table geometry; discard markup and preserve cell text only."""

    def __init__(self) -> None:
        super().__init__()
        self.cells: list[TableCell] = []
        self.row = -1
        self.column = 0
        self.occupied: set[tuple[int, int]] = set()
        self.current: dict[str, Any] | None = None
        self.text: list[str] = []
        self.in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self.in_head = True
        elif tag == "tr":
            self.row += 1
            self.column = 0
        elif tag in {"td", "th"}:
            if self.current is not None or self.row < 0:
                raise ValueError("Malformed table cell structure")
            while (self.row, self.column) in self.occupied:
                self.column += 1
            attributes = dict(attrs)
            self.current = {
                "row": self.row,
                "column": self.column,
                "row_span": int(attributes.get("rowspan") or "1"),
                "column_span": int(attributes.get("colspan") or "1"),
                "is_header": tag == "th" or self.in_head,
            }
            self.text = []
        elif tag == "br" and self.current is not None:
            self.text.append("\n")

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self.in_head = False
        if tag not in {"td", "th"} or self.current is None:
            return
        cell = TableCell(**self.current, text="".join(self.text).strip())
        for row in range(cell.row, cell.row + cell.row_span):
            for column in range(cell.column, cell.column + cell.column_span):
                if (row, column) in self.occupied:
                    raise ValueError("Overlapping table cells")
                self.occupied.add((row, column))
        self.cells.append(cell)
        self.column += cell.column_span
        self.current = None


def table_cells(html: str) -> tuple[int, int, list[TableCell]]:
    parser = TableHTMLParser()
    parser.feed(html)
    parser.close()
    if not parser.cells or parser.current is not None:
        raise ValueError("No complete HTML table cells found")
    return (
        max(cell.row + cell.row_span for cell in parser.cells),
        max(cell.column + cell.column_span for cell in parser.cells),
        parser.cells,
    )
