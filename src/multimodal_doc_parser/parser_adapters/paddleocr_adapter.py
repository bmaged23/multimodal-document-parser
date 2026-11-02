"""Normalize saved PaddleOCR-VL page results without importing GPU libraries."""

import json
import re
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any

from PIL import Image

from multimodal_doc_parser.document_schema.document import (
    AssetReference,
    BoundingBox,
    ChartElement,
    Document,
    DocumentElement,
    FormulaElement,
    HeadingElement,
    ImageElement,
    Provenance,
    SourceRegion,
    TableElement,
    TextElement,
)
from multimodal_doc_parser.parser_adapters.document_structure import finalize_structure
from multimodal_doc_parser.parser_adapters.table_html import table_cells


def normalize_paddleocr_results(
    document: Document,
    bundle: Path,
    raw_directory: Path,
    *,
    model_version: str = "1.6",
    allow_partial: bool = False,
) -> tuple[Document, list[str]]:
    """Use rendered-page coordinates; fail rather than silently omit unsupported blocks."""
    elements: list[DocumentElement] = []
    headings: list[HeadingElement] = []
    warnings: list[str] = []
    pending_pages: list[int] = []
    provenance = Provenance(
        producer="PaddleOCR-VL", model_version=model_version, method="extracted"
    )
    text_labels = {
        "text": "text",
        "content": "text",
        "abstract": "text",
        "reference": "text",
        "list": "list_item",
        "list_item": "list_item",
        "figure_title": "caption",
        "table_title": "caption",
        "chart_title": "caption",
        "caption": "caption",
        "vision_footnote": "caption",
        "aside_text": "text",
        "number": "page_footer",
        "formula_number": "text",
        "algorithm": "text",
        "footnote": "footnote",
        "header": "page_header",
        "footer": "page_footer",
        "page_number": "page_footer",
        "reference_content": "text",
    }
    for page in document.pages:
        path = raw_directory / f"page_{page.page_number:04d}.json"
        if allow_partial and not path.exists():
            pending_pages.append(page.page_number)
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        raw = saved.get("res", saved)
        if raw.get("width") != page.width_pixels or raw.get("height") != page.height_pixels:
            raise ValueError(f"Page {page.page_number}: raw dimensions do not match rendered page")
        if "parsing_res_list" not in raw:
            raise ValueError(f"Page {page.page_number}: missing parsing_res_list")
        for block in raw["parsing_res_list"]:
            identity = f"element_{len(elements) + 1:05d}"
            coordinates = [float(value) for value in block["block_bbox"]]
            if len(coordinates) != 4 or not all(isfinite(value) for value in coordinates):
                raise ValueError("Expected a four-coordinate PaddleOCR block_bbox")
            normalized = [
                coordinates[0] / page.width_pixels,
                coordinates[1] / page.height_pixels,
                coordinates[2] / page.width_pixels,
                coordinates[3] / page.height_pixels,
            ]
            clipped = [max(0, min(1, value)) for value in normalized]
            if clipped != normalized:
                warnings.append(f"{identity}: bounding box clipped to page bounds")
            box = BoundingBox(x0=clipped[0], y0=clipped[1], x1=clipped[2], y1=clipped[3])
            common: dict[str, Any] = {
                "id": identity,
                "parent_id": headings[-1].id if headings else None,
                "source_regions": [SourceRegion(page_number=page.page_number, bounding_box=box)],
                "provenance": provenance,
            }
            label = str(block["block_label"]).lower()
            content = str(block.get("block_content") or "")
            element: DocumentElement
            if label in {"doc_title", "paragraph_title", "title", "section_title"}:
                hashes = re.match(r"^(#{1,6})\s+", content)
                level = len(hashes[1]) if hashes else (1 if label == "doc_title" else 2)
                warnings.append(f"{identity}: heading level inferred from label/Markdown")
                while headings and headings[-1].level >= level:
                    headings.pop()
                common["parent_id"] = headings[-1].id if headings else None
                heading = HeadingElement(
                    **common, text=re.sub(r"^#{1,6}\s+", "", content), level=level
                )
                headings.append(heading)
                element = heading
            elif label == "table":
                rows, columns, cells = table_cells(content)
                element = TableElement(**common, row_count=rows, column_count=columns, cells=cells)
            elif label in {"formula", "display_formula", "inline_formula", "equation"}:
                element = FormulaElement(**common, latex=content)
            elif label in {"image", "figure", "picture", "chart"}:
                relative = f"assets/crops/{identity}.png"
                destination = bundle / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(bundle / page.image.path) as image:
                    crop_box = (
                        floor(box.x0 * image.width),
                        floor(box.y0 * image.height),
                        ceil(box.x1 * image.width),
                        ceil(box.y1 * image.height),
                    )
                    with image.crop(crop_box) as crop:
                        crop.save(destination)
                asset = AssetReference(path=relative)
                element = (
                    ChartElement(**common, asset=asset)
                    if label == "chart"
                    else ImageElement(**common, asset=asset)
                )
                if content:
                    warnings.append(
                        f"{identity}: visual recognition content retained in raw result"
                    )
            else:
                kind = text_labels.get(label, "text")
                if label not in text_labels:
                    warnings.append(f"{identity}: unsupported label {label!r} represented as text")
                element = TextElement.model_validate({**common, "type": kind, "text": content})
            elements.append(element)
    payload = document.model_dump()
    payload.update(
        status="parsing" if pending_pages else "parsed",
        elements=[element.model_dump() for element in elements],
    )
    normalized_document = Document.model_validate(payload)
    if not pending_pages:
        normalized_document = finalize_structure(normalized_document, bundle, warnings)
    return normalized_document, warnings
