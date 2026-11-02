"""Verify the complete saved-result import, normalization, and export path."""

import json
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document, TableElement
from multimodal_doc_parser.parser_adapters.parsing_run import (
    ParsingRunError,
    parse_prepared_document,
)
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document


@pytest.fixture
def parsed_run(tmp_path: Path) -> Path:
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 250, "Synthetic test")
    canvas.showPage()
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    return parse_prepared_document(
        prepared,
        tmp_path / "parsed",
        Path("unused-runtime"),
        raw_results_directory=Path("tests/fixtures/paddleocr"),
    )


def test_normalization_and_exports(parsed_run: Path) -> None:
    document = Document.model_validate_json((parsed_run / "document.json").read_text())
    assert document.status == "parsed"
    assert len(document.elements) == 8
    assert document.elements[1].parent_id == document.elements[0].id
    assert document.elements[2].parent_id == document.elements[1].id
    assert document.elements[2].source_regions[0].bounding_box.x0 == 0.05
    table = document.elements[3]
    assert isinstance(table, TableElement)
    assert table.cells[0].row_span == 2
    with Image.open(parsed_run / "assets/crops/element_00005.png") as image:
        assert image.size == (90, 60)
    assert (parsed_run / "raw/paddleocr/page_0001.json").exists()
    markdown = (parsed_run / "document.md").read_text()
    assert 'rowspan="2"' in markdown
    assert "source: element_00003; pages: 1" in markdown
    html = (parsed_run / "review.html").read_text()
    assert 'src="source.pdf#page=1"' in html
    assert 'data-page="1"' in html
    assert "<script>window.injected" not in html
    assert "&lt;script&gt;window.injected" in html
    assert "@@TITLE@@" in html  # Input cannot trigger a second template replacement.
    status = json.loads((parsed_run / "run-status.json").read_text())
    assert status["status"] == "completed"
    assert status["mode"] == "import"
    summary = json.loads((parsed_run / "run-summary.json").read_text())
    assert summary["element_counts"]["table"] == 1
    assert summary["table_cells"] == 3
    assert summary["inference"] is None


def test_failed_import_retains_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 20, "Test")
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    raw = tmp_path / "bad-results"
    raw.mkdir()
    (raw / "page_0001.json").write_text(json.dumps({"width": 1, "height": 1}))
    destination = tmp_path / "failed"
    with pytest.raises(ParsingRunError, match="dimensions"):
        parse_prepared_document(prepared, destination, Path("unused"), raw_results_directory=raw)
    assert json.loads((destination / "run-status.json").read_text())["status"] == "failed"
    assert not (destination / "review.html").exists()
    assert Document.model_validate_json((destination / "document.json").read_text()).status == (
        "prepared"
    )
    with pytest.raises(ParsingRunError, match="already exists"):
        parse_prepared_document(prepared, destination, Path("unused"), raw_results_directory=raw)


def test_launcher_preserves_isolated_virtual_environment(tmp_path: Path) -> None:
    """A venv interpreter symlink must not be resolved to its base interpreter."""
    import venv

    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 20, "Test")
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    runtime = tmp_path / "test-runtime"
    venv.EnvBuilder(with_pip=False).create(runtime / ".venv")
    # A lightweight stand-in writes fixture data, never invokes a real model.
    script = (
        "import json, sys\nfrom pathlib import Path\n"
        "assert sys.prefix != sys.base_prefix, 'Lost virtual environment'\n"
        "raw = Path(sys.argv[2]); raw.mkdir(parents=True)\n"
        f"data = {Path('tests/fixtures/paddleocr/page_0001.json').read_text()!r}\n"
        "(raw / 'page_0001.json').write_text(data)\n"
    )
    (runtime / "run_inference.py").write_text(script)
    destination = parse_prepared_document(prepared, tmp_path / "parsed", runtime)
    assert json.loads((destination / "run-status.json").read_text())["status"] == "completed"
