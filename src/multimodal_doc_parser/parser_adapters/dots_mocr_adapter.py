"""Normalize dots.mocr layout responses with explicit input-image coordinates."""

import json
import re
from math import ceil, floor
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from multimodal_doc_parser.document_schema.document import (
    AssetReference,
    BoundingBox,
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
from multimodal_doc_parser.parser_adapters.table_html import table_cells


class LayoutCell(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bbox: tuple[float, float, float, float]
    category: str = Field(min_length=1)
    text: str = ""


def decode_layout(content: str) -> list[dict[str, Any]]:
    """Accept JSON arrays or an explicit cells wrapper; never repair truncated JSON."""
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("\n```"):
        content = content[8:-4]
    decoded = json.loads(content)
    cells = decoded.get("cells") if isinstance(decoded, dict) else decoded
    if not isinstance(cells, list):
        raise ValueError("Expected a dots.mocr layout array or a cells object")
    return [LayoutCell.model_validate(cell).model_dump() for cell in cells]


def normalize_dots_mocr_results(
    document: Document,
    bundle: Path,
    raw_directory: Path,
    *,
    model_version: str,
    allow_partial: bool = False,
) -> tuple[Document, list[str]]:
    elements: list[DocumentElement] = []
    headings: list[HeadingElement] = []
    notes: list[str] = []
    missing_pages: list[int] = []
    provenance = Provenance(producer="dots.mocr", model_version=model_version, method="extracted")
    text_types = {
        "Text": "text",
        "Caption": "caption",
        "Footnote": "footnote",
        "List-item": "list_item",
        "Page-header": "page_header",
        "Page-footer": "page_footer",
    }
    for page in document.pages:
        path = raw_directory / f"page_{page.page_number:04d}.json"
        if allow_partial and not path.exists():
            missing_pages.append(page.page_number)
            notes.append(
                f"Page {page.page_number}: no accepted extraction; inspect page diagnostics"
            )
            continue
        raw = json.loads(path.read_text())
        if (
            raw.get("format_version") != "dots-mocr-page-v1"
            or raw.get("page_number") != page.page_number
            or raw.get("source_width") != page.width_pixels
            or raw.get("source_height") != page.height_pixels
            or raw.get("coordinate_space") != "input_pixels"
        ):
            raise ValueError(f"Page {page.page_number}: incompatible raw identity or dimensions")
        width, height = raw["input_width"], raw["input_height"]
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError("Invalid dots.mocr input dimensions")
        if raw.get("finish_reason") != "stop":
            raise ValueError("Incomplete dots.mocr generation; raw response retained")
        # Decode the preserved model response, rather than trusting a second edited cell copy.
        cells = decode_layout(raw["content"])
        for cell in cells:
            identity = f"element_{len(elements) + 1:05d}"
            values = [v / scale for v, scale in zip(cell["bbox"], [width, height] * 2, strict=True)]
            clipped = [max(0.0, min(1.0, v)) for v in values]
            if values != clipped:
                notes.append(f"{identity}: bounding box clipped to page bounds")
            box = BoundingBox(x0=clipped[0], y0=clipped[1], x1=clipped[2], y1=clipped[3])
            common: dict[str, Any] = {
                "id": identity,
                "parent_id": headings[-1].id if headings else None,
                "source_regions": [SourceRegion(page_number=page.page_number, bounding_box=box)],
                "provenance": provenance,
            }
            label, content = cell["category"], cell["text"]
            element: DocumentElement
            if label in {"Title", "Section-header"}:
                hashes = re.match(r"^(#{1,6})\s+", content)
                level = len(hashes[1]) if hashes else (1 if label == "Title" else 2)
                notes.append(f"{identity}: heading level inferred from category/Markdown")
                while headings and headings[-1].level >= level:
                    headings.pop()
                common["parent_id"] = headings[-1].id if headings else None
                heading = HeadingElement(
                    **common, text=re.sub(r"^#{1,6}\s+", "", content), level=level
                )
                headings.append(heading)
                element = heading
            elif label == "Table":
                rows, columns, cells_grid = table_cells(content)
                element = TableElement(
                    **common, row_count=rows, column_count=columns, cells=cells_grid
                )
            elif label == "Formula":
                element = FormulaElement(**common, latex=content)
            elif label == "Picture":
                relative = f"assets/crops/{identity}.png"
                destination = bundle / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(bundle / page.image.path) as image:
                    with image.crop(
                        (
                            floor(box.x0 * image.width),
                            floor(box.y0 * image.height),
                            ceil(box.x1 * image.width),
                            ceil(box.y1 * image.height),
                        )
                    ) as crop:
                        crop.save(destination)
                element = ImageElement(**common, asset=AssetReference(path=relative))
                if content:
                    notes.append(f"{identity}: picture content retained in raw response")
            elif label in text_types:
                element = TextElement.model_validate(
                    {**common, "type": text_types[label], "text": content}
                )
            else:
                raise ValueError(f"Unsupported dots.mocr category: {label!r}")
            elements.append(element)
    payload = document.model_dump()
    payload.update(
        status="parsing" if missing_pages else "parsed",
        elements=[element.model_dump() for element in elements],
    )
    normalized_document = Document.model_validate(payload)
    if not missing_pages:
        from multimodal_doc_parser.parser_adapters.document_structure import finalize_structure

        normalized_document = finalize_structure(normalized_document, bundle, notes)
    return normalized_document, notes
