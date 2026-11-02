"""Structure inference must keep values intact and make heuristic decisions explicit."""

from copy import deepcopy

from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_exports.readable_exports import inline_source_html
from multimodal_doc_parser.document_schema.document import (
    Document,
    ImageElement,
    TableElement,
)
from multimodal_doc_parser.parser_adapters.document_structure import (
    associate_tables,
    infer_headers,
    recover_pdf_images,
)
from multimodal_doc_parser.parser_adapters.table_html import table_cells
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document
from multimodal_doc_parser.review_interface.html_report import render_markdown_preview


def region(page=1, y0=0.2, y1=0.6):
    return [dict(page_number=page, bounding_box=dict(x0=0.1, y0=y0, x1=0.9, y1=y1))]


def table(html, identity="table", page=1):
    rows, columns, cells = table_cells(html)
    return TableElement(
        id=identity,
        source_regions=region(page),
        provenance=dict(producer="synthetic", method="extracted"),
        row_count=rows,
        column_count=columns,
        cells=cells,
    )


def test_merged_header_inference_preserves_data_and_distinguishes_inference():
    item = table(
        '<tr><td rowspan="2">Country</td><td colspan="2">Accounts</td></tr>'
        "<tr><td>Year</td><td>Amount</td></tr>"
        "<tr><td>A</td><td>2024</td><td>-0.050</td></tr>"
    )
    data = [(c.row, c.column, c.text, c.row_span, c.column_span) for c in item.cells]
    infer_headers(item)
    assert all(c.is_header and c.header_inferred for c in item.cells if c.row < 2)
    assert not any(c.is_header for c in item.cells if c.row == 2)
    assert item.header_inference
    assert data == [(c.row, c.column, c.text, c.row_span, c.column_span) for c in item.cells]
    infer_headers(item)
    assert data == [(c.row, c.column, c.text, c.row_span, c.column_span) for c in item.cells]


def test_explicit_headers_and_ambiguous_text_lists_are_not_reclassified():
    explicit = table("<thead><tr><td>A</td><td>B</td></tr></thead><tr><td>2024</td><td>1</td></tr>")
    infer_headers(explicit)
    assert explicit.cells[0].is_header and not explicit.cells[0].header_inferred
    ambiguous = table("<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr>")
    infer_headers(ambiguous)
    assert not any(c.is_header for c in ambiguous.cells)


def test_continuations_retain_alternating_column_groups_and_note_sources(tmp_path):
    path = tmp_path / "source.pdf"
    canvas = Canvas(str(path))
    for _ in range(3):
        canvas.drawString(50, 700, "Synthetic table page")
        canvas.showPage()
    canvas.save()
    bundle = prepare_document(path, tmp_path / "prepared")
    document = Document.model_validate_json((bundle / "document.json").read_text())
    document.status = "parsed"
    for p, group in [(1, "Accounts"), (2, "Finance"), (3, "Accounts")]:
        caption = dict(
            id=f"caption{p}",
            type="caption",
            text="Table A. Records" + (" (continued)" if p > 1 else ""),
            source_regions=region(p, 0.15, 0.18),
            provenance=dict(producer="fixture", method="extracted"),
        )
        from multimodal_doc_parser.document_schema.document import TextElement

        document.elements.append(TextElement.model_validate(caption))
        document.elements.append(
            table(
                f'<tr><td>Country</td><td colspan="2">{group}</td></tr>'
                "<tr><td>A</td><td>2024</td><td>1</td></tr>",
                identity=f"table{p}",
                page=p,
            )
        )
    document.elements.append(
        TextElement(
            id="note",
            type="caption",
            text="Note: units are USD.",
            source_regions=region(3, 0.62, 0.65),
            provenance=dict(producer="fixture", method="extracted"),
        )
    )
    associate_tables(document)
    tables = [e for e in document.elements if isinstance(e, TableElement)]
    assert len({t.logical_table_id for t in tables}) == 1
    assert [t.continuation_index for t in tables] == [1, 2, 3]
    assert [t.column_group for t in tables] == ["Accounts", "Finance", "Accounts"]
    assert all(t.footnote_element_ids == ["note"] for t in tables)
    assert [t.caption_element_id for t in tables] == ["caption1", "caption2", "caption3"]
    Document.model_validate(document.model_dump())
    broken = deepcopy(document.model_dump())
    broken["elements"][1]["caption_element_id"] = "missing"
    import pytest

    with pytest.raises(ValueError, match="caption"):
        Document.model_validate(broken)


def test_large_native_photo_recovery_is_idempotent_and_skips_ocr_covered_scans(tmp_path):
    image = Image.new("RGB", (180, 220), "navy")
    for x in range(90):
        for y in range(220):
            image.putpixel((x, y), (x * 2, y, 120))
    image.save(tmp_path / "photo.png")
    canvas = Canvas(str(tmp_path / "photo.pdf"), pagesize=(200, 300))
    canvas.drawImage(str(tmp_path / "photo.png"), 10, 40, width=180, height=220)
    canvas.save()
    bundle = prepare_document(tmp_path / "photo.pdf", tmp_path / "prepared", dpi=72)
    document = Document.model_validate_json((bundle / "document.json").read_text())
    document.status = "parsed"
    warnings = []
    recover_pdf_images(document, bundle, warnings)
    assert len(document.elements) == 1 and isinstance(document.elements[0], ImageElement)
    assert document.elements[0].provenance.producer == "PDF native image recovery"
    assert (bundle / document.elements[0].asset.path).exists()
    recover_pdf_images(document, bundle, warnings)
    assert len(document.elements) == 1
    scan = deepcopy(document.model_dump())
    scan["elements"] = [
        dict(
            id="body",
            type="text",
            text="Dense OCR content",
            source_regions=region(1, 0.1, 0.9),
            provenance=dict(producer="fixture", method="extracted"),
        )
    ]
    document = Document.model_validate(scan)
    recover_pdf_images(document, bundle, [])
    assert len(document.elements) == 1


def test_footnote_rendering_is_narrow_safe_and_survives_markdown_sanitization():
    html = inline_source_html("Source $ ^{1} $ and $ Source^{{2}} $ <script>bad()</script>")
    assert "<sup>1</sup>" in html and "Source<sup>2</sup>" in html
    assert "<script>" not in html
    assert "<sup>1</sup>" in render_markdown_preview(html)
    assert inline_source_html("$E=mc^2$") == "$E=mc^2$"
