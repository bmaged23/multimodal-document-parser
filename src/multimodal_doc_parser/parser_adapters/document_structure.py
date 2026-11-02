"""Infer conservative table structure and recover unrepresented native PDF images."""

import re
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image, ImageStat

from multimodal_doc_parser.document_schema.document import (
    AssetReference,
    BoundingBox,
    ChartElement,
    Document,
    ImageElement,
    Provenance,
    SourceRegion,
    TableElement,
    TextElement,
)


def overlap(a: BoundingBox, b: BoundingBox) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def infer_headers(table: TableElement) -> None:
    """Respect explicit headers; infer only an initial header band with layout evidence."""
    if any(c.is_header for c in table.cells):
        return
    first = [c for c in table.cells if c.row == 0]
    nonempty = [c for c in first if c.text.strip()]
    if not nonempty:
        return

    def numeric(text: str) -> bool:
        return bool(re.fullmatch(r"[\d\s.,%+−–/()\-]+", text.strip()))

    rows = 0
    # A spanning title alone might be a category row rather than a column header.
    if len(nonempty) >= 2 and any(c.column_span > 1 or c.row_span > 1 for c in first):
        rows = min(4, max(c.row_span for c in first))
        following = [c for c in table.cells if c.row == rows]
        if rows == 1 and following and not any(numeric(c.text) for c in following if c.text):
            rows += 1
    elif first and not first[0].text.strip() and len(nonempty) >= 2:
        rows = 1
    elif len(nonempty) >= 2 and not any(numeric(c.text) for c in nonempty):
        following = [c for c in table.cells if c.row == 1 and c.text.strip()]
        if following and sum(numeric(c.text) for c in following) >= len(following) / 2:
            rows = 1
    if rows:
        for cell in table.cells:
            if cell.row < rows and cell.text.strip():
                cell.is_header = True
                cell.header_inferred = True
        table.header_inference = (
            f"Initial {rows} row(s) inferred from merged bands/labels and data transition; "
            "heuristic, not a verified semantic header map."
        )


def associate_tables(document: Document) -> None:
    tables = [e for e in document.elements if isinstance(e, TableElement)]
    previous: dict[str, TableElement] = {}
    for table in tables:
        infer_headers(table)
        region = table.source_regions[0]
        box = region.bounding_box
        candidates = []
        for element in document.elements:
            if not isinstance(element, TextElement) or element.type != "caption":
                continue
            if element.source_regions[0].page_number != region.page_number:
                continue
            other = element.source_regions[0].bounding_box
            if (
                re.match(r"^Table\s+\S+", element.text, re.I)
                and 0 <= box.y0 - other.y1 < 0.12
                and min(box.x1, other.x1) > max(box.x0, other.x0)
            ):
                candidates.append((box.y0 - other.y1, element))
        if candidates:
            _, caption = min(candidates, key=lambda item: item[0])
            table.caption = caption.text
            table.caption_element_id = caption.id
        for element in document.elements:
            if not isinstance(element, TextElement):
                continue
            other_region = element.source_regions[0]
            other = other_region.bounding_box
            if (
                other_region.page_number == region.page_number
                and 0 <= other.y0 - box.y1 < 0.20
                and (
                    element.type == "footnote"
                    or re.match(
                        r"^(?:Notes?\s*:|\d{1,2}(?=\s|[A-Z])|\$\s*\^)", element.text.lstrip()
                    )
                )
            ):
                table.footnote_element_ids.append(element.id)
        title = re.sub(r"\s*\(continued\)\s*", "", table.caption or "", flags=re.I)
        title = re.sub(r"\s+", " ", title).strip().casefold()
        prior = previous.get(title) if title else None
        if (
            prior
            and prior.source_regions[0].page_number + 1 == region.page_number
            and "continued" in (table.caption or "").casefold()
        ):
            table.logical_table_id = prior.logical_table_id
            table.continuation_index = (prior.continuation_index or 1) + 1
        else:
            table.logical_table_id = f"table_group_{table.id}"
            table.continuation_index = 1
        if title:
            previous[title] = table
        groups = [c.text for c in table.cells if c.row == 0 and c.column_span > 1]
        table.column_group = " | ".join(groups) or None
    # Notes on the last continuation page apply to the logical table, while
    # keeping their actual source element/page identities.
    for table in tables:
        table.footnote_element_ids = list(
            dict.fromkeys(
                identity
                for fragment in tables
                if fragment.logical_table_id == table.logical_table_id
                for identity in fragment.footnote_element_ids
            )
        )


def recover_pdf_images(document: Document, bundle: Path, warnings: list[str]) -> None:
    source = bundle / "source.pdf"
    if not source.exists():
        return
    recovered: list[ImageElement] = []
    with pdfium.PdfDocument(source) as pdf:
        for info in document.pages:
            page = pdf[info.page_number - 1]
            try:
                if page.get_rotation() != 0:
                    warnings.append(
                        f"Page {info.page_number}: native image recovery skipped rotation"
                    )
                    continue
                width, height = page.get_size()
                occupied = [
                    e
                    for e in document.elements + recovered
                    if e.source_regions[0].page_number == info.page_number
                ]
                for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
                    x0, y0, x1, y1 = image.get_bounds()
                    coords = [
                        max(0.0, x0 / width),
                        max(0.0, 1 - y1 / height),
                        min(1.0, x1 / width),
                        min(1.0, 1 - y0 / height),
                    ]
                    if coords[2] <= coords[0] or coords[3] <= coords[1]:
                        continue
                    box = BoundingBox(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])
                    area = (box.x1 - box.x0) * (box.y1 - box.y0)
                    if area < 0.15:
                        continue
                    visual_coverage = sum(
                        overlap(box, e.source_regions[0].bounding_box)
                        for e in occupied
                        if isinstance(e, (ImageElement, ChartElement, TableElement))
                    )
                    text_coverage = sum(
                        overlap(box, e.source_regions[0].bounding_box)
                        for e in occupied
                        if isinstance(e, TextElement)
                    )
                    if visual_coverage / area >= 0.6 or text_coverage / area > 0.35:
                        continue
                    serial = len(recovered) + 1
                    used_ids = {e.id for e in document.elements + recovered}
                    while f"recovered_image_{serial:05d}" in used_ids:
                        serial += 1
                    identity = f"recovered_image_{serial:05d}"
                    relative = f"assets/crops/{identity}.png"
                    with Image.open(bundle / info.image.path) as rendered:
                        with rendered.crop(
                            (
                                int(box.x0 * rendered.width),
                                int(box.y0 * rendered.height),
                                int(box.x1 * rendered.width),
                                int(box.y1 * rendered.height),
                            )
                        ) as crop:
                            # Flat backgrounds are not meaningful visual candidates.
                            with crop.convert("RGB") as rgb:
                                if max(ImageStat.Stat(rgb).stddev) < 12:
                                    continue
                            (bundle / relative).parent.mkdir(parents=True, exist_ok=True)
                            crop.save(bundle / relative)
                    element = ImageElement(
                        id=identity,
                        source_regions=[
                            SourceRegion(page_number=info.page_number, bounding_box=box)
                        ],
                        provenance=Provenance(
                            producer="PDF native image recovery", method="extracted"
                        ),
                        asset=AssetReference(path=relative),
                    )
                    recovered.append(element)
                    occupied.append(element)
                    warnings.append(
                        f"{identity}: uncovered PDF image recovered on page {info.page_number}; "
                        "classification requires visual enrichment"
                    )
            finally:
                page.close()
    # Existing extraction order/IDs stay intact. Insert recovered images within
    # their source page, before the first element whose top is below their center.
    for element in recovered:
        region = element.source_regions[0]
        center = (region.bounding_box.y0 + region.bounding_box.y1) / 2
        index = next(
            (
                i
                for i, existing in enumerate(document.elements)
                if existing.source_regions[0].page_number > region.page_number
                or (
                    existing.source_regions[0].page_number == region.page_number
                    and existing.source_regions[0].bounding_box.y0 > center
                )
            ),
            len(document.elements),
        )
        document.elements.insert(index, element)


def finalize_structure(document: Document, bundle: Path, warnings: list[str]) -> Document:
    associate_tables(document)
    recover_pdf_images(document, bundle, warnings)
    return Document.model_validate(document.model_dump())
