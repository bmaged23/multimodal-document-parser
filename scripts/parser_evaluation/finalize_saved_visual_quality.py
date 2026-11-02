"""Apply source-only quality checks to saved model responses without new inference."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import ChartElement, Document, ImageElement
from multimodal_doc_parser.review_interface.html_report import write_html_report
from multimodal_doc_parser.visual_enrichment.source_checks import (
    finalize_visual_source_checks,
    flag_conflicting_chart_values,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output_directory.resolve()
    if destination.exists() or destination.is_relative_to(source):
        parser.error("Use a new destination outside the saved source run")
    document = Document.model_validate_json((source / "document.json").read_text())
    if document.status != "parsed":
        parser.error("Expected a fully parsed input")
    with (source / "source.pdf").open("rb") as stream:
        if document.document_id != "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest():
            parser.error("Source PDF identity mismatch")
    source_digest = hashlib.sha256((source / "document.json").read_bytes()).hexdigest()
    shutil.copytree(source, destination)
    raw = destination / "raw/visual_enrichment"
    for element in document.elements:
        if isinstance(element, (ChartElement, ImageElement)):
            reviewer = element.quality_review.reviewer if element.quality_review else None
            finalize_visual_source_checks(document, element, destination, raw, reviewer)
    flag_conflicting_chart_values(document)
    manifest = json.loads((destination / "enrichment-manifest.json").read_text())
    manifest["postprocessing_version"] = "source-quality-v2"
    manifest["visuals_needing_review"] = [
        e.id
        for e in document.elements
        if isinstance(e, (ChartElement, ImageElement))
        and e.quality_review
        and e.quality_review.status == "needs_review"
    ]
    for record in manifest["visuals"]:
        element = next(e for e in document.elements if e.id == record["element_id"])
        if isinstance(element, (ChartElement, ImageElement)) and element.description:
            record["analysis"] = element.description.analysis.model_dump()
            if element.quality_review:
                record["quality_review"] = element.quality_review.model_dump()
            if isinstance(element, ChartElement) and element.series_provenance:
                record["numeric_provenance"] = element.series_provenance.model_dump()
    manifest["status"] = (
        "completed_with_errors"
        if manifest["failed_visuals"]
        else "completed_with_warnings"
        if manifest["visuals_needing_review"]
        else "completed"
    )
    metadata = dict(
        stage="source_only_postprocessing",
        new_model_calls=0,
        new_gpu_ocr_pages=0,
        source_enriched_run=str(source),
        source_document_json_sha256=source_digest,
        completed_at_utc=datetime.now(UTC).isoformat(),
        pdfium_version=version("pypdfium2"),
        scope="Uses native PDF objects and saved predictions only. No reference annotations read. "
        "Raw draft/review model responses are retained unchanged.",
        source_correlated_charts=[
            e.id
            for e in document.elements
            if isinstance(e, ChartElement)
            and e.quality_review
            and e.quality_review.status == "source_correlated"
        ],
    )
    manifest["source_postprocessing"] = metadata
    (destination / "source-postprocessing-manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (destination / "enrichment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "document.json").write_text(document.model_dump_json(indent=2) + "\n")
    (destination / "document.md").write_text(document_markdown(document))
    notes = json.loads((destination / "enrichment-notes.json").read_text())
    notes.append(
        "Source-only quality pass: no new model inference. Native PDF endpoint values "
        "are distinguished from generated descriptions; inferred headers remain heuristic."
    )
    (destination / "enrichment-notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    write_html_report(document, destination, notes)
    assert hashlib.sha256((source / "document.json").read_bytes()).hexdigest() == source_digest
    print(destination / "review.html")


if __name__ == "__main__":
    main()
