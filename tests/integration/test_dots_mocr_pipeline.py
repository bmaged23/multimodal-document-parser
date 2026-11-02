"""Exercise the second backend without downloading model weights in CI."""

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document, TableElement
from multimodal_doc_parser.parser_adapters import dots_mocr_inference
from multimodal_doc_parser.parser_adapters.parsing_run import (
    ParsingRunError,
    parse_prepared_document,
)
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    pdf = tmp_path / "example.pdf"
    canvas = Canvas(str(pdf), pagesize=(200, 300))
    canvas.drawString(10, 250, "Synthetic report")
    canvas.save()
    return prepare_document(pdf, tmp_path / "prepared", dpi=72)


def test_dots_import_exports_source_geometry_and_provenance(prepared: Path, tmp_path: Path) -> None:
    run = parse_prepared_document(
        prepared,
        tmp_path / "run",
        Path("unused"),
        backend="dots-mocr",
        model_version="test-fixture-Q4",
        raw_results_directory=Path("tests/fixtures/dots_mocr"),
    )
    document = Document.model_validate_json((run / "document.json").read_text())
    assert document.status == "parsed"
    assert document.elements[2].parent_id == document.elements[1].id
    table = document.elements[3]
    assert isinstance(table, TableElement)
    assert table.cells[0].row_span == 2
    assert table.cells[2].text == "12"
    picture = document.elements[4]
    assert picture.source_regions[0].bounding_box.x0 == 0.25
    assert all(element.provenance.producer == "dots.mocr" for element in document.elements)
    assert all(
        element.provenance.model_version == "test-fixture-Q4" for element in document.elements
    )
    from PIL import Image

    with Image.open(run / "assets/crops/element_00005.png") as image:
        assert image.size == (100, 50)
    assert (run / "document.md").exists()
    assert "dots.mocr test-fixture-Q4" in (run / "review.html").read_text()
    assert "<script>alert(1)</script>" not in (run / "review.html").read_text()
    assert json.loads((run / "run-status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("failure", ["truncated", "invalid_json", "timeout", "wrong_model"])
def test_http_failures_keep_diagnostics_without_publishing_report(
    prepared: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    def request(url, timeout, payload=None):
        assert timeout == 12
        if url.endswith("/models"):
            return {"data": [{"id": "wrong" if failure == "wrong_model" else "fixture"}]}
        assert payload["model"] == "fixture"
        assert payload["max_tokens"] == 512
        assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        if failure == "timeout":
            raise TimeoutError("test deadline reached")
        return {
            "choices": [
                {
                    "finish_reason": "length" if failure == "truncated" else "stop",
                    "message": {"content": "broken JSON"},
                }
            ]
        }

    monkeypatch.setattr(dots_mocr_inference, "request_json", request)
    run = tmp_path / "failed"
    with pytest.raises(ParsingRunError):
        parse_prepared_document(
            prepared,
            run,
            Path("unused"),
            backend="dots-mocr",
            model="fixture",
            model_version="fixture",
            timeout_seconds=12,
            max_tokens=512,
        )
    assert not (run / "review.html").exists()
    manifest = json.loads((run / "raw/dots_mocr/inference-manifest.json").read_text())
    assert manifest["completed_pages"] == 0
    assert manifest["status"] == "failed"
    assert json.loads((run / "run-status.json").read_text())["status"] == "failed"
    if failure in {"truncated", "invalid_json"}:
        assert (run / "raw/dots_mocr/response_0001.json").exists()


def test_http_pipeline_preserves_complete_response(prepared, tmp_path, monkeypatch):
    content = '[{"bbox":[0,0,196,280],"category":"Text","text":"Synthetic report"}]'

    def request(url, timeout, payload=None):
        if payload is None:
            return {"data": [{"id": "fixture"}]}
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"completion_tokens": 30},
        }

    monkeypatch.setattr(dots_mocr_inference, "request_json", request)
    run = parse_prepared_document(
        prepared,
        tmp_path / "run",
        Path("unused"),
        backend="dots-mocr",
        model="fixture",
        model_version="fixture",
    )
    summary = json.loads((run / "run-summary.json").read_text())
    assert summary["inference"]["backend"] == "dots.mocr"
    assert summary["inference"]["completed_pages"] == 1
    assert (run / "raw/dots_mocr/input-images/page_0001.png").exists()
    assert (run / "raw/dots_mocr/response_0001.json").exists()


def test_partial_export_keeps_missing_pages_explicit(prepared, tmp_path):
    from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import normalize_dots_mocr_results
    from multimodal_doc_parser.review_interface.html_report import write_html_report

    original = Document.model_validate_json((prepared / "document.json").read_text())
    payload = original.model_dump()
    payload["pages"].append({**payload["pages"][0], "page_number": 2})
    document = Document.model_validate(payload)
    normalized, notes = normalize_dots_mocr_results(
        document,
        prepared,
        Path("tests/fixtures/dots_mocr"),
        model_version="fixture",
        allow_partial=True,
    )
    assert normalized.status == "parsing"
    assert len(normalized.elements) == 7
    assert any("Page 2: no accepted extraction" in note for note in notes)
    (prepared / "run-status.json").write_text(
        json.dumps(
            {
                "status": "completed_with_errors",
                "completed_pages": 1,
                "total_pages": 2,
            }
        )
    )
    report = write_html_report(normalized, prepared, notes).read_text()
    assert "Parsing finished with incomplete output: 1/2 pages accepted." in report
