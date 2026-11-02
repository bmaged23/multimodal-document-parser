"""Conservative source-coordinate associations; proximity is not semantic proof."""

import re
from typing import Any

from multimodal_doc_parser.document_schema.document import (
    ChartElement,
    Document,
    ImageElement,
    TableElement,
    TextElement,
)


def reference_labels(text: str) -> set[str]:
    labels = re.findall(r"\b(figure|fig\.?|table)\s+([A-Za-z]\d*|\d+)(?:\.(\d+))?\b", text, re.I)
    return {
        ("table" if kind.casefold() == "table" else "figure")
        + ":"
        + number.casefold()
        + ("." + suffix if suffix else "")
        for kind, number, suffix in labels
    }


def discover_explanations(document: Document) -> dict[str, Any]:
    anchors = [
        e for e in document.elements if isinstance(e, (TableElement, ChartElement, ImageElement))
    ]
    positions = {e.id: i for i, e in enumerate(document.elements)}
    relations = []
    ambiguities = []
    explicit = set()
    caption_text = {a.id: a.caption or "" for a in anchors}
    for anchor in anchors:
        if isinstance(anchor, TableElement):
            for eid in anchor.footnote_element_ids:
                explicit.add(eid)
                relations.append(
                    dict(
                        anchor_id=anchor.id,
                        text_id=eid,
                        method="explicit_table_footnote",
                        inferred=False,
                    )
                )
            if anchor.caption_element_id:
                explicit.add(anchor.caption_element_id)
    for anchor in anchors:
        if caption_text[anchor.id]:
            continue
        index = positions[anchor.id]
        caption_candidates = []
        for offset in [-1, 1]:
            if not 0 <= index + offset < len(document.elements):
                continue
            candidate = document.elements[index + offset]
            if not isinstance(candidate, TextElement) or candidate.type != "caption":
                continue
            if candidate.parent_id != anchor.parent_id or not re.match(
                r"\s*(?:figure|fig\.?|table)\s+", candidate.text, re.I
            ):
                continue
            near = False
            for a in anchor.source_regions:
                for b in candidate.source_regions:
                    if a.page_number != b.page_number:
                        continue
                    ab, bb = a.bounding_box, b.bounding_box
                    gap = ab.y0 - bb.y1 if offset == -1 else bb.y0 - ab.y1
                    overlap = max(0.0, min(ab.x1, bb.x1) - max(ab.x0, bb.x0))
                    width = min(ab.x1 - ab.x0, bb.x1 - bb.x0)
                    near |= width > 0 and overlap / width >= 0.6 and -0.015 <= gap <= 0.08
            if near:
                caption_candidates.append(candidate)
        if len(caption_candidates) == 1:
            caption_text[anchor.id] = caption_candidates[0].text
            explicit.add(caption_candidates[0].id)
            relations.append(
                dict(
                    anchor_id=anchor.id,
                    text_id=caption_candidates[0].id,
                    method="adjacent_source_caption",
                    role="caption",
                    inferred=True,
                )
            )
    for element in document.elements:
        if not isinstance(element, TextElement) or element.type not in {
            "text",
            "footnote",
            "caption",
        }:
            continue
        if element.id in explicit or not element.text.strip():
            continue
        candidates = []
        labels = reference_labels(element.text)
        for anchor in anchors:
            if anchor.parent_id != element.parent_id:
                continue
            shared_page = any(
                a.page_number == b.page_number
                for a in anchor.source_regions
                for b in element.source_regions
            )
            if not shared_page:
                continue
            if labels and labels.intersection(reference_labels(caption_text[anchor.id])):
                candidates.append((anchor, "explicit_caption_reference"))
        if len(candidates) == 1:
            if (
                element.type == "caption"
                and element.text.strip() == (candidates[0][0].caption or "").strip()
            ):
                continue  # The item's caption is already repeated in its chunks.
            relations.append(
                dict(
                    anchor_id=candidates[0][0].id,
                    text_id=element.id,
                    method=candidates[0][1],
                    inferred=True,
                )
            )
            continue
        if len(candidates) > 1:
            ambiguities.append(
                dict(
                    text_id=element.id,
                    candidate_ids=[a.id for a, _ in candidates],
                    reason="Reference matches multiple items on the same page",
                )
            )
            continue
        nearby = []
        for anchor in anchors:
            if (
                anchor.parent_id != element.parent_id
                or positions[anchor.id] >= positions[element.id]
            ):
                continue
            between = document.elements[positions[anchor.id] + 1 : positions[element.id]]
            # Side-by-side visuals remain competing candidates; reading order
            # must not hide that ambiguity. Never cross body text or headings.
            if any(
                e.type not in {"caption", "page_header", "page_footer", "chart", "image", "table"}
                for e in between
            ):
                continue
            for a in anchor.source_regions:
                for b in element.source_regions:
                    if a.page_number != b.page_number:
                        continue
                    box, text_box = a.bounding_box, b.bounding_box
                    gap = text_box.y0 - box.y1
                    overlap = max(0.0, min(box.x1, text_box.x1) - max(box.x0, text_box.x0))
                    width = min(box.x1 - box.x0, text_box.x1 - text_box.x0)
                    cue = bool(
                        re.match(
                            r"\s*(?:notes?|source|this\s+(?:chart|figure|table|image)|"
                            r"the\s+(?:chart|figure|table|image)|values|error bars)\b",
                            element.text,
                            re.I,
                        )
                    )
                    if (
                        width > 0
                        and overlap / width >= 0.6
                        and 0 <= gap <= (0.06 if cue else 0.015)
                    ):
                        nearby.append((anchor.id, gap))
        identities = sorted({identity for identity, _ in nearby})
        if len(identities) == 1:
            relations.append(
                dict(
                    anchor_id=identities[0],
                    text_id=element.id,
                    method="adjacent_below_same_section_and_column",
                    inferred=True,
                )
            )
        elif len(identities) > 1:
            ambiguities.append(
                dict(
                    text_id=element.id,
                    candidate_ids=identities,
                    reason="Below-text proximity is ambiguous",
                )
            )
    return dict(
        relations=relations,
        ambiguous=ambiguities,
        policy=(
            "Explicit source links first; otherwise unique same-page/section reference "
            "or immediately adjacent below text with horizontal overlap. "
            "Ambiguous candidates remain standalone."
        ),
    )
