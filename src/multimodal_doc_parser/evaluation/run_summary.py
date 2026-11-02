"""Operational extraction measurements; deliberately not document accuracy scores."""

import json
from collections import Counter
from pathlib import Path

from multimodal_doc_parser.document_schema.document import Document, TableElement, TextElement


def write_run_summary(
    document: Document, bundle: Path, warnings: list[str], *, raw_directory: Path | None = None
) -> Path:
    manifest = (raw_directory or bundle / "raw" / "paddleocr") / "inference-manifest.json"
    summary = {
        "measurement_scope": "operational; no annotated ground truth scored",
        "document_id": document.document_id,
        "pages": len(document.pages),
        "element_counts": dict(Counter(element.type for element in document.elements)),
        "text_characters": sum(
            len(element.text) for element in document.elements if isinstance(element, TextElement)
        ),
        "table_cells": sum(
            len(element.cells) for element in document.elements if isinstance(element, TableElement)
        ),
        "normalization_note_count": len(warnings),
        "inference": json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.exists()
        else None,
    }
    destination = bundle / "run-summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return destination
