"""Comparison must reject unmatched inputs and distinguish agreement from accuracy."""

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.evaluation.parser_comparison import compare_parser_runs, edit_distance
from multimodal_doc_parser.parser_adapters.parsing_run import parse_prepared_document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document


def test_matched_runs_and_reference_scoring(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 250, "Synthetic report")
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    left = parse_prepared_document(
        prepared,
        tmp_path / "left",
        Path("unused"),
        raw_results_directory=Path("tests/fixtures/paddleocr"),
    )
    right = parse_prepared_document(
        prepared,
        tmp_path / "right",
        Path("unused"),
        backend="dots-mocr",
        model_version="synthetic-fixture",
        raw_results_directory=Path("tests/fixtures/dots_mocr"),
    )
    output = compare_parser_runs(left, right, tmp_path / "comparison")
    report = json.loads((output / "comparison.json").read_text())
    assert report["quality_evaluation"]["status"] == "not_scored"
    assert report["quality_evaluation"]["aggregate"] == []
    assert report["per_page"][0]["parsers"][0]["page_inference_seconds"] is None
    assert report["per_page"][0]["text_agreement_ratio"] < 1
    assert (output / "comparison.html").exists()
    with pytest.raises(ValueError, match="already exists"):
        compare_parser_runs(left, right, output)
    annotation = tmp_path / "reference.json"
    annotation.write_text(
        json.dumps(
            {
                "document_id": report["document_id"],
                "annotation_provenance": "Synthetic test-only reference; not an accuracy benchmark",
                "page_text": {
                    "1": "Synthetic report Results Revenue rose. Year Revenue 12 E=mc^2 1"
                },
            }
        )
    )
    scored = compare_parser_runs(left, right, tmp_path / "scored", reference_path=annotation)
    quality = json.loads((scored / "comparison.json").read_text())["quality_evaluation"]
    assert quality["status"] == "scored_reference_text"
    assert len(quality["aggregate"]) == 2
    assert all(item["character_error_rate"] >= 0 for item in quality["aggregate"])
    document_path = right / "document.json"
    document = json.loads(document_path.read_text())
    document["preparation"]["dpi"] = 144
    document_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="same source"):
        compare_parser_runs(left, right, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


def test_edit_distance_counts_insertions_deletions_and_substitutions():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(["one", "two"], ["one", "three"]) == 1


def test_table_cell_checks_are_scoped_and_missing_cells_fail(tmp_path):
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 250, "Synthetic report")
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    run = parse_prepared_document(
        prepared,
        tmp_path / "run",
        Path("unused"),
        backend="dots-mocr",
        model_version="fixture",
        raw_results_directory=Path("tests/fixtures/dots_mocr"),
    )
    annotation = tmp_path / "reference.json"
    annotation.write_text(
        json.dumps(
            {
                "document_id": json.loads((run / "document.json").read_text())["document_id"],
                "annotation_provenance": "Synthetic test fixture, not a model benchmark",
                "table_cells": [
                    {
                        "page_number": 1,
                        "table_index": 0,
                        "row": 0,
                        "column": 0,
                        "text": "Year",
                        "row_span": 2,
                    },
                    {"page_number": 1, "table_index": 0, "row": 1, "column": 1, "text": "wrong"},
                    {"page_number": 1, "table_index": 0, "row": 99, "column": 0, "text": "missing"},
                ],
            }
        )
    )
    output = compare_parser_runs(run, run, tmp_path / "comparison", reference_path=annotation)
    quality = json.loads((output / "comparison.json").read_text())["quality_evaluation"]
    assert quality["status"] == "scored_table_checks"
    assert quality["aggregate"] == []
    assert quality["table_cell_check_summary"][0] == {"parser_index": 0, "passed": 1, "checked": 3}
