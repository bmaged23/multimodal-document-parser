"""Deterministic adaptive chunking without OCR, LLM calls or source mutation."""

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal, cast

from multimodal_doc_parser.document_schema.document import (
    ChartElement,
    Document,
    DocumentElement,
    FormulaElement,
    HeadingElement,
    ImageElement,
    TableElement,
    TextElement,
)
from multimodal_doc_parser.rag_preparation.chunk_schema import (
    CellReference,
    ChildRecord,
    ChunkSettings,
    EvidencePart,
    ParentRecord,
    PartSlice,
)
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def identity(*values: object) -> str:
    encoded = json.dumps(values, sort_keys=True, default=lambda v: v.model_dump(mode="json"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def heading_elements(
    element: DocumentElement, lookup: dict[str, DocumentElement]
) -> list[HeadingElement]:
    result = []
    parent_id = element.parent_id
    while parent_id:
        parent = lookup[parent_id]
        if not isinstance(parent, HeadingElement):
            break
        result.append(parent)
        parent_id = parent.parent_id
    return list(reversed(result))


def part(element: DocumentElement, role: str, text: str, **kwargs: Any) -> EvidencePart:
    return EvidencePart.model_validate(
        dict(
            id=f"{element.id}:{role}:{identity(text, kwargs)}",
            role=role,
            text=text,
            source_element_ids=[element.id],
            provenance=[element.provenance],
            **kwargs,
        )
    )


def table_parts(table: TableElement, lookup: dict[str, DocumentElement]) -> list[EvidencePart]:
    result = []
    for linked_id, role in [
        (table.caption_element_id, "caption"),
        *[(i, "footnote") for i in table.footnote_element_ids],
    ]:
        linked = lookup.get(linked_id or "")
        if isinstance(linked, TextElement):
            item = part(linked, role, linked.text, context_only=True)
            item.source_element_ids.insert(0, table.id)
            item.id = table.id + ":" + item.id
            result.append(item)
    if table.caption and not table.caption_element_id:
        result.append(part(table, "caption", table.caption, context_only=True))
    header_end = 0
    while header_end < table.row_count:
        cells = [c for c in table.cells if c.row <= header_end < c.row + c.row_span]
        if not cells or not all(c.is_header for c in cells):
            break
        header_end += 1
    headers = [c for c in table.cells if c.row < header_end]
    paths = {}
    for column in range(table.column_count):
        names = [
            c.text
            for c in sorted(headers, key=lambda c: c.row)
            if c.column <= column < c.column + c.column_span
        ]
        paths[column] = " / ".join(dict.fromkeys(names)) or f"column {column + 1}"
    header_text = "Column paths: " + "; ".join(f"[{c + 1}] {paths[c]}" for c in paths)
    result.append(
        part(
            table,
            "table_header",
            header_text,
            context_only=True,
            cells=[CellReference(element_id=table.id, row=c.row, column=c.column) for c in headers],
            inferred=any(c.header_inferred for c in headers),
            column_group=table.column_group,
        )
    )
    for row in range(header_end, table.row_count):
        cells = sorted(
            [c for c in table.cells if c.row <= row < c.row + c.row_span and c.row >= header_end],
            key=lambda c: c.column,
        )
        text = f"Source row {row + 1}: " + " | ".join(
            f"{paths[c.column]}: {c.text}"
            + (f" [spans {c.column_span} columns]" if c.column_span > 1 else "")
            for c in cells
        )
        result.append(
            part(
                table,
                "table_row",
                text,
                cells=[
                    CellReference(element_id=table.id, row=c.row, column=c.column) for c in cells
                ],
                column_group=table.column_group,
            )
        )
    # A header-only table must remain searchable and count as represented.
    if header_end == table.row_count:
        result[-1].context_only = False
    return result


def visual_parts(element: ChartElement | ImageElement) -> list[EvidencePart]:
    result = []
    if element.caption:
        result.append(part(element, "caption", element.caption, context_only=True))
    if element.description:
        description = part(element, "description", element.description.text)
        description.provenance = [element.description.provenance]
        result.append(description)
        analysis = element.description.analysis
        if analysis:
            metadata = analysis.model_dump(exclude={"series", "description"})
            extra = part(element, "visual_analysis", json.dumps(metadata, ensure_ascii=False))
            extra.provenance = [element.description.provenance]
            result.append(extra)
        series = (
            element.series
            if isinstance(element, ChartElement) and element.series
            else (analysis.series if analysis else [])
        )
        for line in series:
            for point in line.points:
                item = part(
                    element,
                    "chart_series",
                    json.dumps(
                        dict(series=line.name, unit=line.unit, **point.model_dump()),
                        ensure_ascii=False,
                    ),
                )
                item.provenance = (
                    [element.series_provenance]
                    if isinstance(element, ChartElement) and element.series_provenance
                    else [element.description.provenance]
                )
                result.append(item)
    elif isinstance(element, ChartElement) and element.series:
        for line in element.series:
            item = part(element, "chart_series", line.model_dump_json())
            item.provenance = [element.series_provenance or element.provenance]
            result.append(item)
    else:
        result.append(
            part(element, "visual_placeholder", "Visual region; description unavailable.")
        )
    return result


def context_for(parent: ParentRecord, slices: list[PartSlice]) -> list[EvidencePart]:
    parts = {p.id: p for p in parent.parts}
    sources = {s for item in slices for s in parts[item.part_id].source_element_ids}
    selected = {s.part_id for s in slices}
    return [
        p
        for p in parent.parts
        if p.context_only
        and p.id not in selected
        and (p.role == "heading" or sources.intersection(p.source_element_ids))
    ]


def render_slices(parent: ParentRecord, slices: list[PartSlice], contexts: list[str]) -> str:
    parts = {p.id: p for p in parent.parts}

    def labelled(p: EvidencePart, text: str) -> str:
        method = ",".join(sorted({v.method for v in p.provenance}))
        flags = "; inferred structure" if p.inferred else ""
        return f"[{p.role}; {method}{flags}]\n{text}"

    chunks = [labelled(parts[p], parts[p].text) for p in contexts]
    chunks.extend(
        labelled(parts[s.part_id], parts[s.part_id].text[s.start : s.end]) for s in slices
    )
    return "\n\n".join(chunks)


def make_children(
    parent: ParentRecord, settings: ChunkSettings, budget: TokenBudget
) -> list[ChildRecord]:
    lookup = {e.id: e for e in parent.source_elements}
    parts = {p.id: p for p in parent.parts}
    result: list[ChildRecord] = []
    pending: list[PartSlice] = []

    def materialize(slices: list[PartSlice]) -> tuple[str, list[str], list[str]]:
        context = context_for(parent, slices)
        included: list[str] = []
        deferred: list[str] = []
        for item in context:
            if (
                budget.count(render_slices(parent, slices, included + [item.id]))
                <= settings.child_max_tokens
            ):
                included.append(item.id)
            else:
                deferred.append(item.id)
        return render_slices(parent, slices, included), included, deferred

    def flush() -> None:
        if not pending:
            return
        text, included, deferred = materialize(pending)
        source_ids = list(
            dict.fromkeys(
                s
                for pid in included + [p.part_id for p in pending]
                for s in parts[pid].source_element_ids
            )
        )
        regions = list(
            {
                r.model_dump_json(): r
                for source in source_ids
                for r in lookup[source].source_regions
            }.values()
        )
        warnings = list(parent.quality_warnings)
        if deferred:
            warnings.append(
                "Some context is deferred to the parent to respect the child token limit."
            )
        if any(p.start or p.end < len(parts[p.part_id].text) for p in pending):
            warnings.append(
                "Oversized evidence part split losslessly; inspect adjacent children "
                "for complete context."
            )
        result.append(
            ChildRecord(
                id="child:" + identity(parent.id, [s.model_dump() for s in pending]),
                parent_id=parent.id,
                document_id=parent.document_id,
                heading_path=parent.heading_path,
                kind=parent.kind,
                text=text,
                token_count=budget.count(text),
                slices=list(pending),
                context_part_ids=included,
                deferred_context_part_ids=deferred,
                source_element_ids=source_ids,
                source_regions=regions,
                column_group=parts[pending[0].part_id].column_group,
                quality_warnings=warnings,
            )
        )
        pending.clear()

    for item in parent.parts:
        if item.context_only or not item.text:
            continue
        # Reserve space for role/provenance labels. Context is optional in the
        # search representation and is always available through parent links.
        for start, end in budget.spans(item.text, settings.child_max_tokens - 40):
            piece = PartSlice(part_id=item.id, start=start, end=end)
            candidate = pending + [piece]
            text, _, _ = materialize(candidate)
            incompatible = pending and (
                parts[pending[-1].part_id].column_group != item.column_group
                or (
                    parent.kind == "table"
                    and parts[pending[-1].part_id].source_element_ids[0]
                    != item.source_element_ids[0]
                )
            )
            if pending and (incompatible or budget.count(text) > settings.child_target_tokens):
                flush()
            pending.append(piece)
    flush()
    # Deferred context must remain independently searchable.
    indexed_context = {pid for child in result for pid in child.context_part_ids}
    for item in parent.parts:
        if item.context_only and item.id not in indexed_context and item.text:
            for start, end in budget.spans(item.text, settings.child_max_tokens - 40):
                pending.append(PartSlice(part_id=item.id, start=start, end=end))
                flush()
    for index, child in enumerate(result):
        child.previous_child_id = result[index - 1].id if index else None
        child.next_child_id = result[index + 1].id if index + 1 < len(result) else None
    parent.child_ids = [c.id for c in result]
    return result


def build_chunks(
    document: Document, settings: ChunkSettings, budget: TokenBudget
) -> tuple[list[ParentRecord], list[ChildRecord], list[dict[str, str]]]:
    if document.status != "parsed":
        raise ValueError("RAG preparation requires a completed parsed document")
    lookup = {e.id: e for e in document.elements}
    fingerprint = identity(document.model_dump(mode="json"), settings.model_dump())
    groups: list[tuple[str, list[DocumentElement]]] = []
    logical_groups: dict[str, list[DocumentElement]] = {}
    exclusions = []
    attached = {
        i
        for e in document.elements
        if isinstance(e, TableElement)
        for i in [e.caption_element_id, *e.footnote_element_ids]
        if i
    }
    text_group: list[DocumentElement] = []
    text_size = 0

    def flush_text() -> None:
        nonlocal text_size
        if text_group:
            groups.append(("text", list(text_group)))
            text_group.clear()
        text_size = 0

    for element in document.elements:
        if isinstance(element, HeadingElement):
            flush_text()
            continue
        if isinstance(element, TextElement) and element.type in {"page_header", "page_footer"}:
            exclusions.append(dict(element_id=element.id, reason="repeated_page_furniture"))
            continue
        if element.id in attached:
            continue
        if isinstance(element, TableElement):
            flush_text()
            key = element.logical_table_id or element.id
            if key not in logical_groups:
                logical_groups[key] = []
                groups.append(("table", logical_groups[key]))
            logical_groups[key].append(element)
        elif isinstance(element, (ImageElement, ChartElement)):
            flush_text()
            groups.append(("visual", [element]))
        else:
            text = element.latex if isinstance(element, FormulaElement) else element.text
            if not text:
                exclusions.append(dict(element_id=element.id, reason="empty_text"))
                continue
            size = budget.count(text)
            if text_group and (
                text_group[-1].parent_id != element.parent_id
                or text_size + size > settings.parent_max_tokens
            ):
                flush_text()
            text_group.append(element)
            text_size += size
    flush_text()
    # Keep headings with no descendant content searchable as standalone records.
    represented_headings = {
        h.id for _, group in groups for e in group for h in heading_elements(e, lookup)
    }
    for element in document.elements:
        if isinstance(element, HeadingElement) and element.id not in represented_headings:
            groups.append(("text", [element]))
    parents = []
    children = []
    for kind, elements in groups:
        headings = heading_elements(elements[0], lookup)
        parts = [part(h, "heading", h.text, context_only=True) for h in headings]
        warnings = []
        if not headings:
            warnings.append("No source heading parent is available; hierarchy was not invented.")
        for element in elements:
            if isinstance(element, TableElement):
                parts.extend(table_parts(element, lookup))
                if element.header_inference:
                    warnings.append("Table header roles are inferred: " + element.header_inference)
                if not any(c.is_header for c in element.cells):
                    warnings.append(
                        "Table has no identified header band; column positions are retained."
                    )
            elif isinstance(element, (ImageElement, ChartElement)):
                parts.extend(visual_parts(element))
                if element.quality_review:
                    warnings.extend(element.quality_review.warnings)
                    warnings.append("Visual quality status: " + element.quality_review.status)
            elif isinstance(element, FormulaElement):
                parts.append(part(element, "formula", element.latex))
            else:
                role = (
                    "heading"
                    if isinstance(element, HeadingElement)
                    else (element.type if element.type in {"caption", "footnote"} else "text")
                )
                parts.append(part(element, role, element.text))
        # Independent identical predictions remain separate source occurrences.
        for index, item in enumerate(parts):
            item.id += f":{index}"
        source_ids = list(dict.fromkeys(i for p in parts for i in p.source_element_ids))
        parent = ParentRecord(
            id="parent:" + identity(fingerprint, kind, [e.id for e in elements]),
            document_id=document.document_id,
            section_id=elements[0].parent_id,
            heading_path=[h.text for h in headings],
            kind=cast(Literal["text", "table", "visual"], kind),
            logical_table_id=elements[0].logical_table_id
            if isinstance(elements[0], TableElement)
            else None,
            source_elements=[lookup[i] for i in source_ids],
            parts=parts,
            quality_warnings=list(dict.fromkeys(warnings)),
        )
        parents.append(parent)
        children.extend(make_children(parent, settings, budget))
    return parents, children, exclusions


def validate_chunks(
    document: Document,
    parents: list[ParentRecord],
    children: list[ChildRecord],
    exclusions: list[dict[str, str]],
    settings: ChunkSettings,
    budget: TokenBudget,
    *,
    oversized_child_ids: set[str] | None = None,
) -> dict[str, Any]:
    lookup = {p.id: p for p in parents}
    if len(lookup) != len(parents) or len({c.id for c in children}) != len(children):
        raise ValueError("Duplicate parent/child identity")
    spans: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for child in children:
        parent = lookup[child.parent_id]
        if child.document_id != document.document_id or child.id not in parent.child_ids:
            raise ValueError("Broken child-to-parent document link")
        if render_slices(parent, child.slices, child.context_part_ids) != child.text:
            raise ValueError("Child content does not match parent evidence slices")
        if child.token_count != budget.count(child.text) or (
            child.token_count > settings.child_max_tokens
            and child.id not in (oversized_child_ids or set())
        ):
            raise ValueError("Child token budget violated")
        for piece in child.slices:
            spans[(parent.id, piece.part_id)].append((piece.start, piece.end))
    for parent in parents:
        actual = [c.id for c in children if c.parent_id == parent.id]
        if actual != parent.child_ids:
            raise ValueError("Parent child ordering mismatch")
        for item in parent.parts:
            if item.context_only:
                continue
            cursor = 0
            for start, end in sorted(spans[(parent.id, item.id)]):
                if start != cursor or end > len(item.text):
                    raise ValueError("Missing, duplicated or invalid source evidence slice")
                cursor = end
            if cursor != len(item.text):
                raise ValueError("Evidence was dropped from child chunks")
        for element in parent.source_elements:
            if isinstance(element, TableElement):
                expected_cells = {(c.row, c.column) for c in element.cells}
                covered_cells = {
                    (c.row, c.column)
                    for p in parent.parts
                    for c in p.cells
                    if c.element_id == element.id
                }
                if expected_cells != covered_cells:
                    raise ValueError("Table cell references were lost or invented")
        searchable = {s.part_id for c in children if c.parent_id == parent.id for s in c.slices}
        searchable.update(
            pid for c in children if c.parent_id == parent.id for pid in c.context_part_ids
        )
        if any(p.text and p.id not in searchable for p in parent.parts):
            raise ValueError("Parent evidence is unavailable to search")
    represented = {e.id for p in parents for e in p.source_elements}
    missing = (
        set(e.id for e in document.elements) - represented - {e["element_id"] for e in exclusions}
    )
    if missing:
        raise ValueError("Unaccounted source elements: " + ", ".join(sorted(missing)))
    return dict(
        status="passed",
        parents=len(parents),
        children=len(children),
        represented_elements=len(represented),
        excluded_elements=exclusions,
        max_child_tokens=max((c.token_count for c in children), default=0),
        source_elements_accounted_for=len(document.elements),
        checks=[
            "source_element_coverage",
            "lossless_body_slice_coverage",
            "parent_child_integrity",
            "exact_token_budgets",
            "rendered_evidence_identity",
            "table_cell_reference_coverage",
            "searchable_context_coverage",
        ],
    )
