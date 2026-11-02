"""A failed page must not discard checkpoints or prevent the next GPU request."""

import importlib.util
import json
import shutil
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document


def test_failed_page_is_preserved_and_next_page_can_succeed(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "batch", "scripts/run_quantized_evaluation_batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    lock = tmp_path / "backends/dots_mocr_runtime/runtime-lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"model_version": "fixture-q4", "server_model_alias": "fixture"}))
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    for text in ["first source page", "second source page"]:
        canvas.drawString(10, 250, text)
        canvas.showPage()
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    parent = Document.model_validate_json((prepared / "document.json").read_text())
    run = tmp_path / "outputs/parser_runs/case"
    shutil.copytree(prepared, run)
    (run / "raw/dots_mocr").mkdir(parents=True)
    batch = module.Batch("fixture")
    case = dict(
        id="case",
        name="fixture",
        run_directory=str(run),
        total_pages=2,
        original_pdf_pages=[5, 7],
        pages=[],
        completed_pages=0,
        failed_page_count=0,
        attempted_pages=0,
        status="running",
    )
    batch.cases.append(case)
    batch.current = case
    monkeypatch.setattr(batch, "ensure_server", lambda: None)
    calls = []

    def parse(bundle, destination, runtime, **kwargs):
        calls.append(destination)
        destination.mkdir()
        shutil.copyfile(bundle / "source.pdf", destination / "source.pdf")
        raw = destination / "raw/dots_mocr"
        raw.mkdir(parents=True)
        if len(calls) == 1:
            raise ValueError("deliberately malformed table")
        (raw / "page_0001.json").write_text('{"page_number":1,"content":"[]"}')
        (raw / "inference-manifest.json").write_text(
            json.dumps({"pages": [{"page_number": 1, "elapsed_seconds": 1}]})
        )
        return destination

    monkeypatch.setattr(module, "parse_prepared_document", parse)
    batch.page(case, 1, parent)
    batch.page(case, 2, parent)
    assert case["attempted_pages"] == 2
    assert case["completed_pages"] == case["failed_page_count"] == 1
    assert [row["status"] for row in case["pages"]] == ["failed", "accepted"]
    assert (run / "page-jobs/page_0001/run/diagnostic-review.html").exists()
    aggregate = json.loads((run / "raw/dots_mocr/page_0002.json").read_text())
    assert aggregate["page_number"] == 2
    assert aggregate["page_job_provenance"]["parent_document_id"] == parent.document_id
    progress = json.loads(batch.progress.read_text())
    assert progress["attempted_pages"] == 2 and progress["accepted_pages"] == 1
