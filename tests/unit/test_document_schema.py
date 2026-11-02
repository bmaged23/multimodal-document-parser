"""Validate structural invariants rather than parser implementation details."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from multimodal_doc_parser.document_schema.document import (
    AssetReference,
    BoundingBox,
    ChartPoint,
    Document,
)


@pytest.mark.parametrize("path", ["../secret", "/absolute.png", "C:\\file.png", "."])
def test_asset_paths_cannot_escape_bundle(path: str) -> None:
    with pytest.raises(ValidationError):
        AssetReference(path=path)


@pytest.mark.parametrize("edges", [(0.8, 0, 0.2, 1), (0, 0, 2, 1), (0, 0, 1, 0)])
def test_invalid_bounding_boxes(edges: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x0=edges[0], y0=edges[1], x1=edges[2], y1=edges[3])


def test_chart_values_cannot_hide_unknown_or_invent_precision() -> None:
    with pytest.raises(ValidationError):
        ChartPoint(label="2023", value=10, value_basis="unknown")
    with pytest.raises(ValidationError):
        ChartPoint(label="2023", value=None, value_basis="printed_label")
    with pytest.raises(ValidationError):
        ChartPoint(label="2023", value=float("nan"), value_basis="visual_estimate")


def example_document() -> dict:
    return json.loads(Path("docs/reference/example-parsed-document.json").read_text())


def test_example_round_trip_and_schema_snapshot() -> None:
    document = Document.model_validate(example_document())
    assert Document.model_validate_json(document.model_dump_json()) == document
    assert json.loads(Path("schemas/document-v1.schema.json").read_text()) == (
        Document.model_json_schema()
    )


@pytest.mark.parametrize("failure", ["missing_page", "missing_parent", "duplicate_id", "overlap"])
def test_invalid_document_relationships(failure: str) -> None:
    data = example_document()
    if failure == "missing_page":
        data["elements"][1]["source_regions"][0]["page_number"] = 2
    elif failure == "missing_parent":
        data["elements"][1]["parent_id"] = "missing"
    elif failure == "duplicate_id":
        data["elements"][1]["id"] = data["elements"][0]["id"]
    else:
        data["elements"][2]["cells"].append(data["elements"][2]["cells"][0])
    with pytest.raises(ValidationError):
        Document.model_validate(data)
