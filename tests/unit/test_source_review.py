"""Evaluation must expose incorrect predictions and reject mismatched evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from multimodal_doc_parser.evaluation.source_review import (
    SourceReviewAnnotations,
    evaluate_source_review,
)


@pytest.fixture
def review_run(tmp_path: Path) -> tuple[Path, Path]:
    source = b"Synthetic source identity for evaluator tests, not a real PDF"
    (tmp_path / "source.pdf").write_bytes(source)
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    region = {"page_number": 1, "bounding_box": dict(x0=0, y0=0, x1=1, y1=1)}
    provenance = dict(producer="synthetic-fixture", method="extracted")
    document = dict(
        document_id=digest,
        source_filename="synthetic.pdf",
        status="parsed",
        preparation=dict(dpi=72, renderer="synthetic-fixture"),
        pages=[
            dict(
                page_number=1,
                width_points=100,
                height_points=100,
                width_pixels=100,
                height_pixels=100,
                rotation_degrees=0,
                image=dict(path="page.png"),
            )
        ],
        elements=[
            dict(
                id="text",
                type="text",
                source_regions=[region],
                provenance=provenance,
                text="First section. Second section.",
            ),
            dict(
                id="table",
                type="table",
                source_regions=[region],
                provenance=provenance,
                row_count=2,
                column_count=2,
                cells=[
                    dict(row=0, column=0, text="Header", column_span=2, is_header=False),
                    dict(row=1, column=0, text="-0.50"),
                ],
            ),
            dict(
                id="chart",
                type="chart",
                source_regions=[region],
                provenance=provenance,
                asset=dict(path="chart.png"),
                description=dict(
                    text="Synthetic chart",
                    provenance=dict(producer="fixture", method="generated"),
                    analysis=dict(
                        kind="chart",
                        description="Synthetic chart",
                        series=[
                            dict(
                                name="A",
                                unit=None,
                                points=[dict(label="2024", value=0.2, value_basis="printed_label")],
                            ),
                            dict(
                                name="B",
                                unit="USD/kWh",
                                points=[dict(label="2024", value=0.1, value_basis="printed_label")],
                            ),
                        ],
                    ),
                ),
            ),
        ],
    )
    (tmp_path / "document.json").write_text(json.dumps(document))
    raw = tmp_path / "raw/paddleocr"
    raw.mkdir(parents=True)
    (raw / "inference-manifest.json").write_text(
        json.dumps(
            dict(
                status="completed",
                completed_pages=1,
                total_pages=1,
                pages=[dict(page_number=1, elapsed_seconds=1.0)],
            )
        )
    )
    annotation = dict(
        document_id=digest,
        annotation_provenance="Synthetic unit test; not real accuracy",
        selection_policy="Selected synthetic failures",
        overview_reviewed_pages=[1],
        table_counts={1: 1},
        visual_counts={1: 1},
        table_cells=[
            dict(
                page_number=1,
                table_index=0,
                row=0,
                column=0,
                text="Header",
                column_span=2,
                is_header=True,
            ),
            dict(page_number=1, table_index=0, row=1, column=0, text="0.50"),
            dict(page_number=1, table_index=0, row=1, column=1, text=""),
        ],
        visuals=[
            dict(
                page_number=1,
                kind="chart",
                all_printed_points_annotated=True,
                printed_points=[
                    dict(series="A", label="2024", value=0.1),
                    dict(series="B", label="2024", value=0.2),
                ],
                series_units={"A": "USD/kWh"},
            )
        ],
        reading_order=[dict(page_number=1, anchors=["First section", "Second section"])],
    )
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(annotation))
    return tmp_path, reference


def test_scoring_separates_values_units_headers_spans_and_missing_cells(review_run):
    run, reference = review_run
    result = evaluate_source_review(run, reference)
    summary = result["summary"]
    assert summary["table_cell_text"] == dict(passed=1, checked=3)
    assert summary["table_cell_span"] == dict(passed=1, checked=1)
    assert summary["table_header_semantics"] == dict(passed=0, checked=1)
    assert summary["printed_chart_point"] == dict(passed=0, checked=2)
    assert summary["predicted_numeric_point_supported"] == dict(passed=0, checked=2)
    assert summary["chart_series_unit"] == dict(passed=0, checked=1)
    assert summary["reading_order_anchor_pair"] == dict(passed=1, checked=1)
    assert "overall_score" not in result


@pytest.mark.parametrize("change", ["document", "source", "unfinished", "outside", "duplicate"])
def test_rejects_invalid_evidence(review_run, change):
    run, reference = review_run
    annotation = json.loads(reference.read_text())
    if change == "document":
        annotation["document_id"] = "sha256:" + "f" * 64
    elif change == "source":
        (run / "source.pdf").write_bytes(b"different source")
    elif change == "unfinished":
        path = run / "document.json"
        document = json.loads(path.read_text())
        document["status"] = "parsing"
        path.write_text(json.dumps(document))
    elif change == "outside":
        annotation["overview_reviewed_pages"] = [2]
    else:
        annotation["table_cells"].append(annotation["table_cells"][0])
    reference.write_text(json.dumps(annotation))
    with pytest.raises(ValueError):
        evaluate_source_review(run, reference)


def test_null_values_abstain_but_axis_ticks_are_not_data_categories(review_run):
    run, reference = review_run
    path = run / "document.json"
    document = json.loads(path.read_text())
    analysis = document["elements"][2]["description"]["analysis"]
    analysis["series"] = [
        dict(name="A", points=[dict(label="0.2", value=None, value_basis="unknown")])
    ]
    path.write_text(json.dumps(document))
    annotation = json.loads(reference.read_text())
    annotation["visuals"] = [
        dict(page_number=1, kind="chart", no_numeric_values=True, no_data_points=True)
    ]
    reference.write_text(json.dumps(annotation))
    summary = evaluate_source_review(run, reference)["summary"]
    assert summary["unlabelled_chart_numeric_abstention"] == dict(passed=1, checked=1)
    assert summary["unlabelled_chart_no_invented_categories"] == dict(passed=0, checked=1)


def test_ambiguous_reading_order_anchor_is_an_annotation_error(review_run):
    run, reference = review_run
    annotation = json.loads(reference.read_text())
    annotation["reading_order"][0]["anchors"] = ["section", "Second section"]
    reference.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate_source_review(run, reference)


def test_duplicate_chart_points_cannot_pass_by_selecting_one_match(review_run):
    run, reference = review_run
    path = run / "document.json"
    document = json.loads(path.read_text())
    series = document["elements"][2]["description"]["analysis"]["series"][0]
    series["points"] = [dict(label="2024", value=0.1, value_basis="printed_label")] * 2
    path.write_text(json.dumps(document))
    checks = evaluate_source_review(run, reference)["checks"]
    point = next(c for c in checks if c["category"] == "printed_chart_point")
    assert not point["passed"]


def test_repository_annotations_validate():
    for path in Path("evaluation/reference_annotations").glob("*-source-review-v1.json"):
        SourceReviewAnnotations.model_validate_json(path.read_text())
