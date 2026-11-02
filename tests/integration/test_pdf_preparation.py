"""Exercise real PDF rendering with small, generated documents."""

from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.pdf_preparation.prepare_document import (
    DocumentPreparationError,
    prepare_document,
)


def create_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=(200, 300))
    canvas.setTitle("Preparation test")
    canvas.drawString(20, 250, "First page")
    canvas.showPage()
    canvas.setPageRotation(90)
    canvas.drawString(20, 100, "Rotated page")
    canvas.showPage()
    canvas.save()


def test_pages_identity_rotation_and_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    create_pdf(source)
    bundle = prepare_document(source, tmp_path / "outputs", dpi=72)
    document = Document.model_validate_json((bundle / "document.json").read_text())
    assert document.title == "Preparation test"
    assert document.status == "prepared"
    assert document.elements == []
    assert len(document.pages) == 2
    assert document.pages[0].width_pixels == 200
    assert document.pages[0].height_pixels == 300
    assert document.pages[1].rotation_degrees == 90
    for page in document.pages:
        with Image.open(bundle / page.image.path) as image:
            assert image.size == (page.width_pixels, page.height_pixels)
            assert image.convert("L").getextrema()[0] < 255  # Content was rendered.
    assert (bundle / "source.pdf").read_bytes() == source.read_bytes()
    assert Document.model_validate_json(document.model_dump_json()) == document
    second = prepare_document(source, tmp_path / "other_outputs", dpi=144)
    other = Document.model_validate_json((second / "document.json").read_text())
    assert other.document_id == document.document_id
    assert other.pages[0].width_pixels == 400
    before = (bundle / "document.json").read_bytes()
    with pytest.raises(DocumentPreparationError, match="already exists"):
        prepare_document(source, tmp_path / "outputs", dpi=72)
    assert (bundle / "document.json").read_bytes() == before
    assert not list((tmp_path / "outputs").glob(".preparing-*"))


@pytest.mark.parametrize("limits", [{"max_pages": 1}, {"max_page_pixels": 10}])
def test_limit_failure_does_not_publish_partial_results(tmp_path: Path, limits: dict) -> None:
    source = tmp_path / "sample.pdf"
    create_pdf(source)
    output = tmp_path / "outputs"
    with pytest.raises(DocumentPreparationError):
        prepare_document(source, output, **limits)
    assert list(output.iterdir()) == []


def test_invalid_and_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(DocumentPreparationError, match="does not exist"):
        prepare_document(tmp_path / "missing.pdf", tmp_path / "outputs")
    source = tmp_path / "invalid.pdf"
    source.write_bytes(b"not a PDF")
    with pytest.raises(DocumentPreparationError, match="Cannot open PDF"):
        prepare_document(source, tmp_path / "outputs")
    assert list((tmp_path / "outputs").iterdir()) == []


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "encrypted.pdf"
    canvas = Canvas(str(source), encrypt="test-password")
    canvas.drawString(20, 20, "Protected")
    canvas.save()
    with pytest.raises(DocumentPreparationError, match="encrypted"):
        prepare_document(source, tmp_path / "outputs")
    assert list((tmp_path / "outputs").iterdir()) == []


def test_later_page_failure_removes_earlier_rendered_assets(tmp_path: Path) -> None:
    source = tmp_path / "oversized-second-page.pdf"
    canvas = Canvas(str(source), pagesize=(100, 100))
    canvas.drawString(10, 50, "Small page")
    canvas.showPage()
    canvas.setPageSize((1000, 1000))
    canvas.drawString(10, 50, "Oversized page")
    canvas.showPage()
    canvas.save()
    output = tmp_path / "outputs"
    with pytest.raises(DocumentPreparationError, match="Page 2 exceeds"):
        prepare_document(source, output, dpi=72, max_page_pixels=20_000)
    assert list(output.iterdir()) == []
