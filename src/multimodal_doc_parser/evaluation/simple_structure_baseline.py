"""Practical section/table/visual chunks with repeated context and no hierarchy retrieval."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
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
from multimodal_doc_parser.evaluation.chunking_baselines import FixedRetriever
from multimodal_doc_parser.evaluation.rag_benchmark import digest, save
from multimodal_doc_parser.rag_preparation.chunk_builder import (
    heading_elements,
    identity,
    part,
    render_slices,
    table_parts,
    validate_chunks,
    visual_parts,
)
from multimodal_doc_parser.rag_preparation.chunk_schema import (
    ChildRecord,
    ChunkSettings,
    ParentRecord,
    PartSlice,
)
from multimodal_doc_parser.rag_preparation.retrieval import verify_bundle
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def basic_table_parts(table: TableElement, lookup: dict[str, DocumentElement]) -> list[Any]:
    """Treat an empty stub in an otherwise labelled header band as header structure.

    Operates on a copy; raw source annotations and the existing production
    serializer remain unchanged. This representation difference is recorded.
    """
    working = table.model_copy(deep=True)
    for row in range(working.row_count):
        cells = [c for c in working.cells if c.row <= row < c.row + c.row_span]
        if not cells or not any(c.is_header and c.text.strip() for c in cells):
            break
        if any(c.text.strip() and not c.is_header for c in cells):
            break
        for cell in cells:
            if not cell.text.strip() and not cell.is_header:
                cell.is_header = True
                cell.header_inferred = True
    result = table_parts(working, lookup)
    # Footnotes remain searchable, in their own table-associated chunks rather
    # than crowding every row group. Captions and column headers always repeat.
    for item in result:
        if item.role == "footnote":
            item.context_only = False
    return result


def pack_standalone(
    parent: ParentRecord, budget: TokenBudget, target: int, maximum: int
) -> list[ChildRecord]:
    """Greedy paragraph/row/point packing; every continuation repeats all context."""
    contexts = [p.id for p in parent.parts if p.context_only]
    if budget.count(render_slices(parent, [], contexts)) >= maximum - 16:
        raise ValueError("Mandatory heading/header context exceeds the configured chunk budget")
    pending: list[PartSlice] = []
    children: list[ChildRecord] = []
    lookup = {e.id: e for e in parent.source_elements}
    parts = {p.id: p for p in parent.parts}

    def tokens(slices: list[PartSlice]) -> int:
        return budget.count(render_slices(parent, slices, contexts))

    def flush() -> None:
        if not pending:
            return
        text = render_slices(parent, pending, contexts)
        ids = list(
            dict.fromkeys(
                e
                for pid in contexts + [s.part_id for s in pending]
                for e in parts[pid].source_element_ids
            )
        )
        child = ChildRecord(
            id="simple-child:" + identity(parent.id, len(children), pending),
            parent_id=parent.id,
            document_id=parent.document_id,
            heading_path=parent.heading_path,
            kind=parent.kind,
            text=text,
            token_count=budget.count(text),
            slices=list(pending),
            context_part_ids=list(contexts),
            deferred_context_part_ids=[],
            source_element_ids=ids,
            source_regions=[r for eid in ids for r in lookup[eid].source_regions],
            quality_warnings=list(parent.quality_warnings),
        )
        children.append(child)
        pending.clear()

    for item in parent.parts:
        if item.context_only or not item.text:
            continue
        whole = PartSlice(part_id=item.id, start=0, end=len(item.text))
        if pending and (tokens(pending) >= target or tokens(pending + [whole]) > maximum):
            flush()
        if tokens([whole]) <= maximum:
            pending.append(whole)
            continue
        # Only an individually oversized paragraph/row/metadata block is split.
        # Context is repeated in full even when one source paragraph is split.
        allowance = maximum - budget.count(render_slices(parent, [], contexts)) - 16
        for start, end in budget.spans(item.text, allowance):
            piece = PartSlice(part_id=item.id, start=start, end=end)
            if tokens([piece]) > maximum:
                raise ValueError("Serialized fragment exceeds the configured budget")
            pending.append(piece)
            flush()
    flush()
    parent.child_ids = [c.id for c in children]
    return children


def build_simple_chunks(
    document: Document, target: int = 350, maximum: int = 500
) -> tuple[list[ParentRecord], list[ChildRecord], dict[str, Any]]:
    if document.status != "parsed" or not 64 <= target <= maximum:
        raise ValueError("Require a parsed document and coherent chunk limits")
    budget = TokenBudget()
    lookup = {e.id: e for e in document.elements}
    attached = {
        i
        for e in document.elements
        if isinstance(e, TableElement)
        for i in [e.caption_element_id, *e.footnote_element_ids]
        if i
    }
    groups: list[tuple[str, list[DocumentElement]]] = []
    text_group: list[DocumentElement] = []
    excluded = []

    def flush_text() -> None:
        if text_group:
            groups.append(("text", list(text_group)))
            text_group.clear()

    for element in document.elements:
        if isinstance(element, HeadingElement):
            flush_text()
            continue
        if isinstance(element, TextElement) and element.type in {"page_header", "page_footer"}:
            excluded.append(dict(element_id=element.id, reason="repeated_page_furniture"))
            continue
        if element.id in attached:
            continue
        if isinstance(element, (TableElement, ChartElement, ImageElement)):
            flush_text()
            groups.append(("table" if isinstance(element, TableElement) else "visual", [element]))
        else:
            value = element.latex if isinstance(element, FormulaElement) else element.text
            if not value:
                excluded.append(dict(element_id=element.id, reason="empty_text"))
                continue
            if text_group and text_group[-1].parent_id != element.parent_id:
                flush_text()
            text_group.append(element)
    flush_text()
    represented = {h.id for _, es in groups for e in es for h in heading_elements(e, lookup)}
    groups.extend(
        ("text", [e])
        for e in document.elements
        if isinstance(e, HeadingElement) and e.id not in represented
    )
    parents = []
    children = []
    for number, (kind, elements) in enumerate(groups):
        headings = heading_elements(elements[0], lookup)
        parts = [part(h, "heading", h.text, context_only=True) for h in headings]
        warnings = []
        for element in elements:
            if isinstance(element, TableElement):
                parts.extend(basic_table_parts(element, lookup))
                warnings.append("Physical table block only; no continued-table joining.")
            elif isinstance(element, (ChartElement, ImageElement)):
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
        for i, item in enumerate(parts):
            item.id += f":simple:{number}:{i}"
        # Header-only tables must still have a searchable body part.
        if not any(not p.context_only and p.text for p in parts):
            parts[-1].context_only = False
        source_ids = list(dict.fromkeys(eid for p in parts for eid in p.source_element_ids))
        parent = ParentRecord(
            id="simple-group:" + identity(document.document_id, number, kind, target, maximum),
            document_id=document.document_id,
            section_id=elements[0].parent_id,
            heading_path=[h.text for h in headings],
            kind=cast(Literal["text", "table", "visual"], kind),
            source_elements=[lookup[eid] for eid in source_ids],
            parts=parts,
            quality_warnings=warnings,
        )
        children.extend(pack_standalone(parent, budget, target, maximum))
        parents.append(parent)
    settings = ChunkSettings(child_target_tokens=target, child_max_tokens=maximum)
    validation = validate_chunks(document, parents, children, excluded, settings, budget)
    for child in children:
        parent = next(p for p in parents if p.id == child.parent_id)
        required = [p.id for p in parent.parts if p.context_only]
        if child.context_part_ids != required:
            raise ValueError("A continuation lost its heading, caption or table header")
    validation["continuation_context_repeated"] = True
    return parents, children, validation


def build_simple_bundles(sources: Sequence[Path], destination: Path) -> list[Path]:
    if destination.exists():
        paths = sorted(p.parent for p in destination.glob("*/chunk_manifest.json"))
        if len(paths) != len(sources):
            raise ValueError("Incomplete simple bundles; preserve and use a new directory")
        for path in paths:
            verify_bundle(path)
        return paths
    bundles = []
    for number, source in enumerate(sources, 1):
        verify_bundle(source)
        path = source / "source_document.json"
        document = Document.model_validate_json(path.read_bytes())
        parents, children, validation = build_simple_chunks(document)
        output = destination / f"document-{number:02}"
        output.mkdir(parents=True)
        record_groups: list[tuple[str, Sequence[ParentRecord | ChildRecord]]] = [
            ("parents", parents),
            ("children", children),
        ]
        for name, records in record_groups:
            (output / (name + ".jsonl")).write_text(
                "".join(r.model_dump_json() + "\n" for r in records)
            )
        save(output / "validation.json", validation)
        save(
            output / "chunk_manifest.json",
            dict(
                document_id=document.document_id,
                algorithm="simple-structure-v1",
                source_document_sha256=digest(path),
                target_tokens=350,
                maximum_tokens=500,
                heading_policy="Repeat the full source heading hierarchy in every continuation",
                tables="Physical tables; repeat caption and column headers; searchable footnotes",
                visuals="One chart/image group; repeat caption/headings when oversized",
                header_policy=(
                    "Empty stubs in otherwise annotated header bands treated as "
                    "inferred header structure; source annotations unchanged"
                ),
                retrieval="Standalone chunks only; no parent expansion or query rewriting",
                artifacts={
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ["parents.jsonl", "children.jsonl", "validation.json"]
                },
            ),
        )
        bundles.append(output)
    return bundles


class SimpleStructureRetriever(FixedRetriever):
    """Serve preassembled standalone chunks, with their actual repeated-context provenance."""

    def expand(self, matched: list[str], maximum: int, expand: bool) -> list[dict[str, Any]]:
        blocks = super().expand(matched, maximum, False)
        for block in blocks:
            child = self.by_id[block["matched_child_ids"][0]]
            block["context_part_ids"] = list(child.context_part_ids)
            block["quality_warnings"] = list(child.quality_warnings)
        return blocks
