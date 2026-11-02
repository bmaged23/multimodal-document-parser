"""Check table structure recovery for merged cells and malformed model output."""

import pytest

from multimodal_doc_parser.parser_adapters.paddleocr_adapter import table_cells


def test_merged_cells_preserve_headers_spans_and_text() -> None:
    rows, columns, cells = table_cells(
        '<table><tr><th colspan="2">A &amp; B</th></tr>'
        "<tr><td>X<br>Y</td><td>18.4</td></tr></table>"
    )
    assert (rows, columns) == (2, 2)
    assert cells[0].is_header
    assert cells[0].column_span == 2
    assert cells[0].text == "A & B"
    assert cells[1].text == "X\nY"


@pytest.mark.parametrize("html", ["no table", "<tr><td>unclosed", "<td>no row</td>"])
def test_incomplete_tables_are_rejected(html: str) -> None:
    with pytest.raises(ValueError):
        table_cells(html)
