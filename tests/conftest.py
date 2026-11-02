"""Shared synthetic inputs for the read-only evaluation showcase."""

import json
import shutil
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document


@pytest.fixture
def showcase_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    pdf = project / "synthetic.pdf"
    canvas = Canvas(str(pdf))
    for number in [1, 2]:
        canvas.drawString(30, 700, f"Synthetic source page {number}")
        canvas.showPage()
    canvas.save()
    prepared = prepare_document(pdf, project / "prepared", dpi=72)
    doc = Document.model_validate_json((prepared / "document.json").read_text())
    doc.status = "parsed"
    doc = Document.model_validate(
        {
            **doc.model_dump(),
            "elements": [
                dict(
                    id=f"body-{number}",
                    type="text",
                    text=f"Page {number}. <script>window.injected=true</script> ![remote](https://example.com/beacon)",
                    source_regions=[
                        dict(page_number=number, bounding_box=dict(x0=0.1, y0=0.1, x1=0.8, y1=0.3))
                    ],
                    provenance=dict(producer="synthetic", method="extracted"),
                )
                for number in [1, 2]
            ],
        }
    )
    for name, pages in [("run-a", [1]), ("run-b", [1, 2])]:
        run = project / "outputs/parser_runs" / name
        shutil.copytree(prepared, run)
        current = doc.model_copy(
            update={
                "elements": [e for e in doc.elements if e.source_regions[0].page_number in pages]
            }
        )
        (run / "document.json").write_text(current.model_dump_json())
        raw = run / "raw/paddleocr"
        raw.mkdir(parents=True)
        (raw / "inference-manifest.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    completed_pages=len(pages),
                    pages=[dict(page_number=n, elapsed_seconds=1.25) for n in pages],
                )
            )
        )
        for n in pages:
            (raw / f"page_{n:04}.json").write_text(
                json.dumps(
                    dict(
                        input_path="/do-not-copy/private.pdf",
                        parsing_res_list=[
                            dict(
                                block_label="text",
                                block_content="Saved synthetic raw block",
                                block_bbox=[1, 2, 3, 4],
                            )
                        ],
                        Authorization="do-not-copy-secret",
                    )
                )
            )

    def run_spec(identity, directory):
        return dict(id=identity, label=identity, stage="ocr", kind="canonical", directory=directory)

    cases = [
        dict(
            id="fixture-public",
            title="Synthetic comparison",
            description="Not model benchmark data",
            group="parsers",
            private=False,
            original_pages=[10, 20],
            source_url=None,
            runs=[
                run_spec("model-a", "run-a"),
                run_spec("model-b", "run-b"),
                run_spec("missing-model", "missing-run"),
            ],
        ),
        dict(
            id="fixture-private",
            title="Private fixture",
            description="Excluded by default",
            group="parsers",
            private=True,
            original_pages=[10, 20],
            source_url=None,
            runs=[run_spec("private-a", "run-a")],
        ),
    ]
    manifest = project / "manifest.json"
    manifest.write_text(json.dumps(dict(version="1.0", cases=cases)))
    return project, manifest
