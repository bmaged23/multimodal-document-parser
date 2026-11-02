"""Compare matched page inputs, separating parser agreement from reference accuracy."""

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any

from multimodal_doc_parser.document_schema.document import (
    Document,
    DocumentElement,
    FormulaElement,
    TableElement,
)
from multimodal_doc_parser.evaluation.reference_annotations import ReferenceAnnotations


def page_text(elements: list[DocumentElement]) -> str:
    parts: list[str] = []
    for element in elements:
        if isinstance(element, TableElement):
            parts.extend(
                cell.text for cell in sorted(element.cells, key=lambda c: (c.row, c.column))
            )
        elif isinstance(element, FormulaElement):
            parts.append(element.latex)
        elif hasattr(element, "text"):
            parts.append(str(element.text))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def load_inference(run: Path) -> dict[str, Any] | None:
    manifests = list((run / "raw").glob("*/inference-manifest.json"))
    if len(manifests) > 1:
        raise ValueError("Ambiguous parser run: multiple inference manifests")
    return json.loads(manifests[0].read_text()) if manifests else None


def compare_parser_runs(
    left_run: Path,
    right_run: Path,
    destination: Path,
    *,
    pages: list[int] | None = None,
    reference_path: Path | None = None,
) -> Path:
    left_run, right_run, destination = (
        left_run.resolve(),
        right_run.resolve(),
        destination.resolve(),
    )
    if any(run in destination.parents for run in [left_run, right_run]):
        raise ValueError("Comparison output must be outside both parser runs")
    if destination.exists():
        raise ValueError(f"Comparison output already exists: {destination}")
    documents = [
        Document.model_validate_json((run / "document.json").read_text())
        for run in [left_run, right_run]
    ]
    left, right = documents
    if (left.document_id, left.preparation, left.pages) != (
        right.document_id,
        right.preparation,
        right.pages,
    ):
        raise ValueError("Comparison requires the same source PDF, rendered pages, and preparation")
    selected = pages if pages is not None else list(range(1, len(left.pages) + 1))
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(number < 1 or number > len(left.pages) for number in selected)
    ):
        raise ValueError("Comparison pages must be unique valid page numbers")
    inferences = [load_inference(run) for run in [left_run, right_run]]
    for doc, inference in zip(documents, inferences, strict=True):
        if doc.status == "prepared":
            raise ValueError("Cannot compare a prepared document")
        if doc.status == "parsing":
            completed = {item["page_number"] for item in (inference or {}).get("pages", [])}
            if not set(selected) <= completed:
                raise ValueError("Selected pages include unfinished parser output")
    references = None
    if reference_path:
        annotated = ReferenceAnnotations.model_validate_json(reference_path.read_text())
        references = annotated.model_dump(mode="json")
        if annotated.document_id != left.document_id:
            raise ValueError("Reference annotations do not match the source PDF")
        annotated_pages = set(annotated.page_text) | {c.page_number for c in annotated.table_cells}
        if not annotated_pages <= set(selected):
            raise ValueError("Reference annotations include pages outside the selected comparison")
    table_results: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for number in selected:
        page_elements = [
            [
                element
                for element in doc.elements
                if any(region.page_number == number for region in element.source_regions)
            ]
            for doc in documents
        ]
        texts = [page_text(elements) for elements in page_elements]
        row = {
            "page_number": number,
            "text_agreement_ratio": SequenceMatcher(
                None, texts[0].split(), texts[1].split(), autojunk=False
            ).ratio(),
            "parsers": [
                {
                    "element_counts": dict(Counter(element.type for element in elements)),
                    "text_characters": len(text),
                    "table_cells": sum(
                        len(e.cells) for e in elements if isinstance(e, TableElement)
                    ),
                    "page_inference_seconds": next(
                        (
                            p.get("elapsed_seconds")
                            for p in (inference or {}).get("pages", [])
                            if p["page_number"] == number
                        ),
                        None,
                    ),
                }
                for elements, text, inference in zip(page_elements, texts, inferences, strict=True)
            ],
        }
        page_rows.append(row)
        for expected_cell in (references or {}).get("table_cells", []):
            if expected_cell["page_number"] != number:
                continue
            for index, elements in enumerate(page_elements):
                tables = [e for e in elements if isinstance(e, TableElement)]
                table_index = expected_cell["table_index"]
                table = tables[table_index] if table_index < len(tables) else None
                cell = (
                    next(
                        (
                            c
                            for c in table.cells
                            if c.row == expected_cell["row"] and c.column == expected_cell["column"]
                        ),
                        None,
                    )
                    if table
                    else None
                )
                passed = (
                    cell is not None
                    and re.sub(r"\s+", " ", cell.text).strip()
                    == re.sub(r"\s+", " ", expected_cell["text"]).strip()
                )
                for span in ["row_span", "column_span"]:
                    if expected_cell.get(span) is not None:
                        passed = (
                            passed
                            and cell is not None
                            and getattr(cell, span) == expected_cell[span]
                        )
                table_results.append(
                    {
                        "parser_index": index,
                        "reference": expected_cell,
                        "actual_text": cell.text if cell else None,
                        "passed": passed,
                    }
                )
        if references and str(number) in references["page_text"]:
            expected = re.sub(r"\s+", " ", references["page_text"][str(number)]).strip()
            for index, actual in enumerate(texts):
                expected_words = expected.split()
                quality.append(
                    {
                        "page_number": number,
                        "parser_index": index,
                        "reference_characters": len(expected),
                        "reference_words": len(expected_words),
                        "character_errors": edit_distance(list(expected), list(actual)),
                        "word_errors": edit_distance(expected_words, actual.split()),
                    }
                )
    report: dict[str, Any] = {
        "comparison_version": "1.0",
        "document_id": left.document_id,
        "pages": selected,
        "scope": (
            "Agreement measures differences, not accuracy. Timing includes backend-specific work; "
            "imports and resumed pages retain their original measurements."
        ),
        "parsers": [
            {
                "run_directory": str(run),
                "producers": sorted(
                    {
                        f"{element.provenance.producer}:{element.provenance.model_version}"
                        for element in doc.elements
                    }
                ),
                "inference": inference,
            }
            for run, doc, inference in zip(
                [left_run, right_run], documents, inferences, strict=True
            )
        ],
        "per_page": page_rows,
        "quality_evaluation": {
            "status": (
                "scored_reference_text" if references["page_text"] else "scored_table_checks"
            )
            if references
            else "not_scored",
            "table_cell_checks": table_results,
            "table_cell_check_summary": [
                {
                    "parser_index": index,
                    "passed": sum(c["passed"] for c in table_results if c["parser_index"] == index),
                    "checked": sum(c["parser_index"] == index for c in table_results),
                }
                for index in range(2)
            ]
            if table_results
            else [],
            "scope": (
                "Whitespace-normalized ordered text, including headings, formulas, "
                "headers/footers, "
                "and table cells in row order; no layout or table-structure accuracy score."
            ),
            "annotation_provenance": references.get("annotation_provenance")
            if references
            else None,
            "per_page_errors": quality,
            "aggregate": [
                {
                    "parser_index": index,
                    "character_error_rate": sum(
                        q["character_errors"] for q in quality if q["parser_index"] == index
                    )
                    / sum(q["reference_characters"] for q in quality if q["parser_index"] == index),
                    "word_error_rate": sum(
                        q["word_errors"] for q in quality if q["parser_index"] == index
                    )
                    / sum(q["reference_words"] for q in quality if q["parser_index"] == index),
                }
                for index in range(2)
            ]
            if quality
            else [],
        },
    }
    for run in [left_run, right_run]:
        if not (run / "review.html").is_file():
            raise ValueError(f"Missing review report: {run}")
    destination.mkdir(parents=True)
    (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    links = [
        escape(os.path.relpath(run / "review.html", destination), quote=True)
        for run in [left_run, right_run]
    ]
    producer_labels = [
        escape(", ".join(item["producers"]) or "No extracted elements")
        for item in report["parsers"]
    ]
    quality_status = escape(report["quality_evaluation"]["status"].replace("_", " "))
    checked_summary = report["quality_evaluation"]["table_cell_check_summary"]
    checked_html = ""
    if checked_summary:
        checked_html = (
            "<p>Source-bound table cell checks: "
            + "; ".join(
                f"{'Left' if item['parser_index'] == 0 else 'Right'} "
                f"{item['passed']}/{item['checked']} passed"
                for item in checked_summary
            )
            + ". These selected checks do not establish overall accuracy.</p>"
        )
    memory_labels = [
        escape(str((inference or {}).get("peak_sampled_device_memory_mib") or "unmeasured"))
        for inference in inferences
    ]

    def seconds(value: float | None) -> str:
        return f"{value:.1f}s" if value is not None else "unmeasured"

    rows = "".join(
        f"<tr><td>{row['page_number']}</td><td>{row['text_agreement_ratio']:.1%}</td>"
        f"<td>{row['parsers'][0]['text_characters']}</td>"
        f"<td>{row['parsers'][1]['text_characters']}</td>"
        f"<td>{seconds(row['parsers'][0]['page_inference_seconds'])}</td>"
        f"<td>{seconds(row['parsers'][1]['page_inference_seconds'])}</td></tr>"
        for row in page_rows
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Parser comparison</title>
<style>body{{font:16px system-ui;margin:20px;background:#f4f6fa;color:#172538;
overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
iframe{{width:100%;height:78vh;border:1px solid #ccd}}
div.metrics{{overflow-x:auto}}
table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Parser comparison</h1><p>Matched source pages. Text agreement is not an accuracy score.
<a href="comparison.json">Full measurements and provenance</a>.</p>
<p>Reference quality: {quality_status}. Timings exclude different work across backends;
review the JSON measurement scope before interpreting speed differences.</p>
{checked_html}
<p>Peak sampled device memory (MiB): Left {memory_labels[0]}; Right {memory_labels[1]}.
See JSON for sampling scope and initialization differences.</p>
<div class="metrics"><table><tr><th>Page</th><th>Text agreement</th><th>Left characters</th>
<th>Right characters</th><th>Left page time</th><th>Right page time</th></tr>{rows}</table></div>
<div class="grid"><section><h2>Left parser</h2><p>{producer_labels[0]}</p>
<a href="{links[0]}">Open review</a>
<iframe title="Left parser review" src="{links[0]}"></iframe></section>
<section><h2>Right parser</h2><p>{producer_labels[1]}</p><a href="{links[1]}">Open review</a>
<iframe title="Right parser review" src="{links[1]}"></iframe></section></div></html>"""
    (destination / "comparison.html").write_text(html, encoding="utf-8")
    return destination
