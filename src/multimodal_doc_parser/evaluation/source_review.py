"""Score explicitly scoped, source-reviewed assertions without modifying predictions."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from multimodal_doc_parser.document_schema.document import (
    ChartElement,
    Document,
    DocumentElement,
    ImageElement,
    NonNegativeInt,
    PositiveInt,
    SchemaModel,
    TableElement,
)
from multimodal_doc_parser.evaluation.reference_annotations import TableCellReference


class ReviewedCell(TableCellReference):
    is_header: bool | None = None


class PrintedPointReference(SchemaModel):
    series: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float


class VisualReference(SchemaModel):
    page_number: PositiveInt
    visual_index: NonNegativeInt = 0
    kind: Literal["chart", "image", "flowchart"]
    printed_points: list[PrintedPointReference] = Field(default_factory=list)
    series_units: dict[str, str] = Field(default_factory=dict)
    all_printed_points_annotated: bool = False
    no_numeric_values: bool = False
    no_data_points: bool = False

    @model_validator(mode="after")
    def consistent_points(self) -> Self:
        keys = [(normalized(p.series), normalized(p.label)) for p in self.printed_points]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate printed-point reference")
        if (self.no_numeric_values or self.no_data_points) and self.printed_points:
            raise ValueError("Abstention references cannot contain numeric points")
        return self


class ReadingOrderReference(SchemaModel):
    page_number: PositiveInt
    anchors: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_anchors(self) -> Self:
        if any(not a.strip() for a in self.anchors) or len(set(self.anchors)) != len(self.anchors):
            raise ValueError("Reading-order anchors must be nonempty and unique")
        return self


class SourceReviewAnnotations(SchemaModel):
    review_version: Literal["1.0"] = "1.0"
    document_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    annotation_provenance: str = Field(min_length=1)
    selection_policy: str = Field(min_length=1)
    overview_reviewed_pages: list[PositiveInt]
    table_counts: dict[int, NonNegativeInt]
    visual_counts: dict[int, NonNegativeInt]
    table_cells: list[ReviewedCell] = Field(default_factory=list)
    visuals: list[VisualReference] = Field(default_factory=list)
    reading_order: list[ReadingOrderReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_targets(self) -> Self:
        cells = [(c.page_number, c.table_index, c.row, c.column) for c in self.table_cells]
        visuals = [(v.page_number, v.visual_index) for v in self.visuals]
        if len(set(cells)) != len(cells) or len(set(visuals)) != len(visuals):
            raise ValueError("Duplicate reference target")
        if not self.overview_reviewed_pages or any(
            n < 1 for n in [*self.table_counts, *self.visual_counts]
        ):
            raise ValueError("Reference page numbers must be positive and scope nonempty")
        return self


def normalized(text: str) -> str:
    """Match label case/spacing and equivalent dash glyphs; keep numbers and signs."""
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s*-\s*", "-", re.sub(r"\s+", " ", text).strip().casefold())


def page_elements(document: Document, page: int) -> list[DocumentElement]:
    return [e for e in document.elements if any(r.page_number == page for r in e.source_regions)]


def evaluate_source_review(run: Path, reference_path: Path) -> dict[str, Any]:
    document = Document.model_validate_json((run / "document.json").read_text())
    reference = SourceReviewAnnotations.model_validate_json(reference_path.read_text())
    if document.document_id != reference.document_id:
        raise ValueError("Reference belongs to a different document")
    with (run / "source.pdf").open("rb") as source:
        source_digest = hashlib.file_digest(source, "sha256").hexdigest()
    if document.document_id != f"sha256:{source_digest}":
        raise ValueError("Run source.pdf hash does not match its canonical document")
    if document.status != "parsed":
        raise ValueError("Quality evaluation requires a completed parsed document")
    pages = {p.page_number: page_elements(document, p.page_number) for p in document.pages}
    selected = set(reference.overview_reviewed_pages) | set(reference.table_counts)
    selected |= set(reference.visual_counts) | {c.page_number for c in reference.table_cells}
    selected |= {v.page_number for v in reference.visuals}
    selected |= {r.page_number for r in reference.reading_order}
    if not selected <= pages.keys():
        raise ValueError("Reference contains a page outside this document")
    checks: list[dict[str, Any]] = []

    def check(category: str, target: dict[str, Any], expected: Any, actual: Any) -> None:
        checks.append(
            dict(
                category=category,
                target=target,
                expected=expected,
                actual=actual,
                passed=expected == actual,
            )
        )

    for kind, counts in [("table", reference.table_counts), ("visual", reference.visual_counts)]:
        for page, count in counts.items():
            actual = sum(
                isinstance(e, TableElement)
                if kind == "table"
                else isinstance(e, (ChartElement, ImageElement))
                for e in pages[page]
            )
            check(f"{kind}_count_per_page", {"page": page}, count, actual)
    for expected in reference.table_cells:
        tables = [e for e in pages[expected.page_number] if isinstance(e, TableElement)]
        table = tables[expected.table_index] if expected.table_index < len(tables) else None
        cell = (
            next(
                (c for c in table.cells if (c.row, c.column) == (expected.row, expected.column)),
                None,
            )
            if table
            else None
        )
        target: dict[str, Any] = dict(
            page=expected.page_number,
            table_index=expected.table_index,
            row=expected.row,
            column=expected.column,
        )

        # Cell text is case-sensitive. Whitespace normalization does not erase signs/decimals.
        def cell_text(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip()

        check(
            "table_cell_text",
            target,
            cell_text(expected.text),
            cell_text(cell.text) if cell else None,
        )
        for name in ["row_span", "column_span", "is_header"]:
            value = getattr(expected, name)
            if value is not None:
                check(
                    "table_header_semantics" if name == "is_header" else "table_cell_span",
                    dict(target, attribute=name),
                    value,
                    getattr(cell, name) if cell else None,
                )
    for expected_visual in reference.visuals:
        visuals = [
            e
            for e in pages[expected_visual.page_number]
            if isinstance(e, (ChartElement, ImageElement))
        ]
        visual = (
            visuals[expected_visual.visual_index]
            if expected_visual.visual_index < len(visuals)
            else None
        )
        analysis = visual.description.analysis if visual and visual.description else None
        target = dict(
            page=expected_visual.page_number,
            visual_index=expected_visual.visual_index,
            element_id=visual.id if visual else None,
        )
        check("visual_kind", target, expected_visual.kind, analysis.kind if analysis else None)
        series = analysis.series if analysis else []
        predicted = [(s.name, s.unit, p) for s in series for p in s.points]
        for point in expected_visual.printed_points:
            matches = [
                p
                for name, _, p in predicted
                if normalized(name) == normalized(point.series)
                and normalized(p.label) == normalized(point.label)
            ]
            matched_point = matches[0] if len(matches) == 1 else None
            passed = (
                matched_point is not None
                and matched_point.value is not None
                and math.isclose(matched_point.value, point.value, rel_tol=0, abs_tol=1e-9)
                and matched_point.value_basis == "printed_label"
            )
            checks.append(
                dict(
                    category="printed_chart_point",
                    target=dict(target, series=point.series, label=point.label),
                    expected=point.value,
                    actual=[p.model_dump() for p in matches],
                    passed=passed,
                )
            )
        for name, unit in expected_visual.series_units.items():
            matching_series = [s for s in series if normalized(s.name) == normalized(name)]
            actual_unit = matching_series[0].unit if len(matching_series) == 1 else None
            check(
                "chart_series_unit",
                dict(target, series=name),
                normalized(unit),
                normalized(actual_unit) if actual_unit is not None else None,
            )
        if expected_visual.all_printed_points_annotated:
            for name, _, predicted_point in predicted:
                if predicted_point.value is None:
                    continue
                supported = any(
                    normalized(p.series) == normalized(name)
                    and normalized(p.label) == normalized(predicted_point.label)
                    and math.isclose(p.value, predicted_point.value, rel_tol=0, abs_tol=1e-9)
                    for p in expected_visual.printed_points
                )
                check(
                    "predicted_numeric_point_supported",
                    dict(
                        target,
                        series=name,
                        label=predicted_point.label,
                        value=predicted_point.value,
                    ),
                    True,
                    supported,
                )
        if expected_visual.no_numeric_values:
            check(
                "unlabelled_chart_numeric_abstention",
                target,
                0,
                sum(p.value is not None for _, _, p in predicted) if analysis else None,
            )
        if expected_visual.no_data_points:
            check(
                "unlabelled_chart_no_invented_categories",
                target,
                0,
                len(predicted) if analysis else None,
            )
    for order in reference.reading_order:
        text = normalized(" ".join(str(getattr(e, "text", "")) for e in pages[order.page_number]))
        if any(text.count(normalized(anchor)) > 1 for anchor in order.anchors):
            raise ValueError("Ambiguous reading-order anchor; use a longer source phrase")
        positions = [text.find(normalized(anchor)) for anchor in order.anchors]
        for index in range(len(positions) - 1):
            before, after = positions[index : index + 2]
            check(
                "reading_order_anchor_pair",
                dict(
                    page=order.page_number,
                    before=order.anchors[index],
                    after=order.anchors[index + 1],
                ),
                True,
                before >= 0 and after > before,
            )
    categories = sorted({c["category"] for c in checks})
    summary = {
        name: {
            "passed": sum(c["passed"] for c in checks if c["category"] == name),
            "checked": sum(c["category"] == name for c in checks),
        }
        for name in categories
    }
    inference = json.loads((run / "raw/paddleocr/inference-manifest.json").read_text())
    return {
        "evaluation_version": "1.1",
        "document_id": document.document_id,
        "run_directory": str(run),
        "annotation_provenance": reference.annotation_provenance,
        "selection_policy": reference.selection_policy,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "document_json_sha256": hashlib.sha256((run / "document.json").read_bytes()).hexdigest(),
        "scope": "Selected assistant source-reviewed development checks; not independent human "
        "ground truth, a held-out benchmark, full CER/WER, or overall accuracy. "
        "No combined score: categories measure different properties.",
        "summary": summary,
        "checks": checks,
        "inventory": dict(
            pages=len(document.pages),
            elements=dict(Counter(e.type for e in document.elements)),
            table_cells=sum(len(e.cells) for e in document.elements if isinstance(e, TableElement)),
        ),
        "normalization_audit": {
            "inferred_header_cells": sum(
                c.header_inferred
                for e in document.elements
                if isinstance(e, TableElement)
                for c in e.cells
            ),
            "tables_with_caption_links": sum(
                bool(e.caption_element_id) for e in document.elements if isinstance(e, TableElement)
            ),
            "tables_with_note_links": sum(
                bool(e.footnote_element_ids)
                for e in document.elements
                if isinstance(e, TableElement)
            ),
            "table_fragments": [
                {
                    "id": e.id,
                    "page": e.source_regions[0].page_number,
                    "logical_table_id": e.logical_table_id,
                    "continuation_index": e.continuation_index,
                    "column_group": e.column_group,
                }
                for e in document.elements
                if isinstance(e, TableElement)
            ],
            "recovered_pdf_images": [
                e.id
                for e in document.elements
                if isinstance(e, ImageElement)
                and e.provenance.producer == "PDF native image recovery"
            ],
        },
        "visual_quality": {
            "statuses": dict(
                Counter(
                    e.quality_review.status
                    for e in document.elements
                    if isinstance(e, (ChartElement, ImageElement)) and e.quality_review
                )
            ),
            "needs_review": [
                e.id
                for e in document.elements
                if isinstance(e, (ChartElement, ImageElement))
                and e.quality_review
                and e.quality_review.status == "needs_review"
            ],
        },
        "runtime": {
            k: inference.get(k)
            for k in [
                "status",
                "completed_pages",
                "total_pages",
                "total_seconds",
                "initialization_seconds",
                "peak_sampled_device_memory_mib",
                "memory_measurement",
                "paddleocr_version",
                "paddlepaddle_version",
            ]
        },
        "page_timings": inference["pages"],
    }
