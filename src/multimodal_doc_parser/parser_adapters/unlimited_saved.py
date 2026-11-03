"""Normalize preserved Unlimited native blocks without generating missing content.

Each page is accepted atomically. Invalid coordinates, incomplete generations or
invalid table grids remain failed pages with their original raw prediction.
"""

import json
import re
import shutil
from math import ceil, floor
from pathlib import Path
from typing import Any

from PIL import Image

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
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
from multimodal_doc_parser.parser_adapters.table_html import table_cells

BLOCK = re.compile(r"<BLOCK>([A-Za-z_][\w-]*)(\[[^\]\n]*\])<FORK>(.*?)(?=<BLOCK>|\Z)", re.S)
PROFILE = (
    "baidu/Unlimited-OCR-2@c14071b549f4fcab10909f3c4acc6a2edb366947; NF4 language / FP16 vision"
)


def normalize_saved_run(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise ValueError("Use a fresh normalized Unlimited run")
    status = json.loads((source / "run-status.json").read_text())
    prepared = Path(status["prepared_bundle"])
    parent = Document.model_validate_json((prepared / "document.json").read_text())
    if parent.document_id != status["document_id"]:
        raise ValueError("Unlimited source identity mismatch")
    shutil.copytree(prepared, destination)
    raw = destination / "raw/unlimited_ocr_2"
    raw.mkdir(parents=True)
    elements: list[DocumentElement] = []
    accepted, failed, notes = [], [], []
    provenance = Provenance(producer="Unlimited-OCR 2", model_version=PROFILE, method="extracted")
    text_types = {
        "header": "page_header",
        "footer": "page_footer",
        "page_number": "page_footer",
        "page_footnote": "footnote",
        "image_footnote": "footnote",
        "image_caption": "caption",
    }
    for page in parent.pages:
        number = page.page_number
        prediction = source / f"page_{number:04d}.json"
        staged: list[DocumentElement] = []
        try:
            value = json.loads(prediction.read_text())
            shutil.copyfile(prediction, raw / prediction.name)
            if value["finish_reason"] != "stop":
                raise ValueError("Incomplete native generation")
            text = value["text"]
            matches = list(BLOCK.finditer(text))
            if not matches or text.count("<BLOCK>") != len(matches) or text.count("<PAGE>") > 1:
                raise ValueError("Unsupported or incomplete native block envelope")
            if text[: matches[0].start()].replace("<PAGE>", "").strip():
                raise ValueError("Unexpected content before native blocks")
            for match in matches:
                category, coordinates, content = match.groups()
                coords = json.loads(coordinates)
                if len(coords) != 4 or any(type(v) not in {int, float} for v in coords):
                    raise ValueError("Native box must contain four numeric coordinates")
                box = BoundingBox(
                    **dict(zip(["x0", "y0", "x1", "y1"], [v / 1000 for v in coords], strict=True))
                )
                identity = f"element_{len(elements) + len(staged) + 1:05d}"
                common: dict[str, Any] = dict(
                    id=identity,
                    source_regions=[SourceRegion(page_number=number, bounding_box=box)],
                    provenance=provenance,
                )
                content = content.strip()
                element: DocumentElement
                if category in {"image", "chart"}:
                    relative = f"assets/crops/{identity}.png"
                    target = destination / relative
                    target.parent.mkdir(exist_ok=True, parents=True)
                    with Image.open(destination / page.image.path) as image:
                        with image.crop(
                            (
                                floor(box.x0 * image.width),
                                floor(box.y0 * image.height),
                                ceil(box.x1 * image.width),
                                ceil(box.y1 * image.height),
                            )
                        ) as crop:
                            crop.save(target)
                    kind = ChartElement if category == "chart" else ImageElement
                    element = kind(
                        **common, asset=AssetReference(path=relative), caption=content or None
                    )
                elif category == "title":
                    element = HeadingElement(**common, text=content, level=2)
                elif category == "equation":
                    element = FormulaElement(**common, latex=content)
                elif category == "table":
                    rows, columns, cells = table_cells(content)
                    element = TableElement(
                        **common, row_count=rows, column_count=columns, cells=cells
                    )
                else:
                    element = TextElement.model_validate(
                        {**common, "type": text_types.get(category, "text"), "text": content}
                    )
                    if category not in text_types and category != "text":
                        notes.append(f"Page {number}: native {category!r} retained as text")
                staged.append(element)
            elements.extend(staged)
            accepted.append(dict(page_number=number, elapsed_seconds=value.get("elapsed_seconds")))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failed.append(dict(page_number=number, error=str(error)))
    document = parent.model_copy(
        update={"status": "parsing" if failed else "parsed", "elements": elements}
    )
    document = Document.model_validate(document.model_dump())
    (destination / "document.json").write_text(document.model_dump_json(indent=2) + "\n")
    (destination / "document.md").write_text(document_markdown(document))
    manifest = dict(
        backend="Unlimited-OCR 2",
        model_version=PROFILE,
        document_id=document.document_id,
        status="completed_with_errors" if failed else "completed",
        completed_pages=len(accepted),
        total_pages=len(document.pages),
        pages=accepted,
        failed_pages=failed,
        reuse_source=str(source),
        notes=(
            "Native boxes use the existing review's 0–1000 convention; "
            "no OCR rerun or inferred missing blocks. "
            "Heading level 2 is inferred from title category."
        ),
    )
    (raw / "inference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "normalization-notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    return destination
