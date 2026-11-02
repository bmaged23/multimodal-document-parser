"""Structural baseline v2: whole semantic units and budgeted related explanations."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multimodal_doc_parser.document_schema.document import Document, TextElement
from multimodal_doc_parser.evaluation.rag_benchmark import digest, save
from multimodal_doc_parser.evaluation.related_explanations import discover_explanations
from multimodal_doc_parser.evaluation.simple_structure_baseline import build_simple_chunks
from multimodal_doc_parser.rag_preparation.chunk_builder import (
    identity,
    part,
    render_slices,
    validate_chunks,
)
from multimodal_doc_parser.rag_preparation.chunk_schema import (
    ChildRecord,
    ChunkSettings,
    ParentRecord,
    PartSlice,
)
from multimodal_doc_parser.rag_preparation.retrieval import verify_bundle
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Conservative paragraph/sentence boundaries, preserving all characters."""
    start = 0
    result = []
    for boundary in re.finditer(r"\n\s*\n|[.!?][\"”’')\]]*\s+(?=[A-Z0-9“\"(])", text):
        prefix = text[: boundary.end()].strip()
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Fig|No|St|vs|e\.g|i\.e)\.$", prefix, re.I):
            continue
        result.append((start, boundary.end()))
        start = boundary.end()
    if start < len(text):
        result.append((start, len(text)))
    return result


def semantic_units(
    parent: ParentRecord, budget: TokenBudget, maximum: int, mandatory: list[str]
) -> list[list[PartSlice]]:
    """Keep rows, JSON records and chart points intact; prefer whole series."""
    result = []
    pending_series: list[PartSlice] = []
    series_name = None

    def fits(slices: list[PartSlice]) -> bool:
        return budget.count(render_slices(parent, slices, mandatory)) <= maximum

    def flush_series() -> None:
        if pending_series:
            if fits(pending_series):
                result.append(list(pending_series))
            else:
                result.extend([[piece] for piece in pending_series])
            pending_series.clear()

    for item in parent.parts:
        if item.context_only or not item.text:
            continue
        whole = PartSlice(part_id=item.id, start=0, end=len(item.text))
        if item.role == "chart_series":
            data = json.loads(item.text)
            name = data.get("series", data.get("name"))
            if pending_series and name != series_name:
                flush_series()
            series_name = name
            pending_series.append(whole)
            continue
        flush_series()
        if fits([whole]) or item.role in {"table_row", "formula", "visual_analysis"}:
            result.append([whole])
        else:
            result.extend(
                [[PartSlice(part_id=item.id, start=a, end=b)] for a, b in sentence_spans(item.text)]
            )
    flush_series()
    return result


def pack_semantic(
    parent: ParentRecord,
    optional: list[str],
    budget: TokenBudget,
    target: int,
    maximum: int,
    note_limit: int,
) -> tuple[list[ChildRecord], set[str]]:
    mandatory = [p.id for p in parent.parts if p.context_only and p.id not in optional]
    parts = {p.id: p for p in parent.parts}
    lookup = {e.id: e for e in parent.source_elements}
    children: list[ChildRecord] = []
    exceptions = set()
    pending: list[PartSlice] = []

    def tokens(slices: list[PartSlice], context: list[str] | None = None) -> int:
        return budget.count(
            render_slices(parent, slices, mandatory if context is None else context)
        )

    def flush() -> None:
        if not pending:
            return
        selected = list(mandatory)
        notes_tokens = 0
        body_ids = {s.part_id for s in pending}
        for pid in optional:
            if pid in body_ids:
                continue
            cost = budget.count(render_slices(parent, [], [pid]))
            if notes_tokens + cost <= note_limit and tokens(pending, selected + [pid]) <= maximum:
                selected.append(pid)
                notes_tokens += cost
        text = render_slices(parent, pending, selected)
        source_ids = list(
            dict.fromkeys(
                eid
                for pid in selected + [s.part_id for s in pending]
                for eid in parts[pid].source_element_ids
            )
        )
        warnings = list(parent.quality_warnings)
        oversized = budget.count(text) > maximum
        if oversized:
            warnings.append(
                "An indivisible semantic unit exceeds the soft token ceiling; preserved intact."
            )
        if any(pid not in selected and pid not in body_ids for pid in optional):
            warnings.append(
                "Related explanation omitted from inline context to preserve primary content; "
                "source text remains searchable."
            )
        child = ChildRecord(
            id="semantic-child:" + identity(parent.id, len(children), pending),
            parent_id=parent.id,
            document_id=parent.document_id,
            heading_path=parent.heading_path,
            kind=parent.kind,
            text=text,
            token_count=budget.count(text),
            slices=list(pending),
            context_part_ids=selected,
            deferred_context_part_ids=[
                pid for pid in optional if pid not in selected and pid not in body_ids
            ],
            source_element_ids=source_ids,
            source_regions=[r for eid in source_ids for r in lookup[eid].source_regions],
            quality_warnings=warnings,
        )
        if oversized:
            exceptions.add(child.id)
        children.append(child)
        pending.clear()

    units = semantic_units(parent, budget, maximum, mandatory)
    complete = [piece for unit in units for piece in unit]
    if parent.kind in {"table", "visual"} and tokens(complete) <= maximum:
        units = [complete]
    for unit in units:
        if pending and (tokens(pending) >= target or tokens(pending + unit) > maximum):
            flush()
        pending.extend(unit)
        if tokens(pending) > maximum:
            # A whole row/record/sentence can exceed the soft ceiling. Never
            # split its bytes just to satisfy that ceiling.
            flush()
    flush()
    parent.child_ids = [c.id for c in children]
    # Optional copies that never fit are represented by their standalone source
    # chunks and explicit relation records, not as inaccessible parent parts.
    used = {pid for c in children for pid in c.context_part_ids}
    removed = {
        p.id for p in parent.parts if p.context_only and p.id in optional and p.id not in used
    }
    parent.parts = [p for p in parent.parts if p.id not in removed]
    for child in children:
        child.deferred_context_part_ids = [
            pid for pid in child.deferred_context_part_ids if pid not in removed
        ]
    return children, exceptions


def build_semantic_chunks(
    document: Document, target: int = 350, maximum: int = 500, note_limit: int = 100
) -> tuple[list[ParentRecord], list[ChildRecord], dict[str, Any], dict[str, Any]]:
    parents, _, base_validation = build_simple_chunks(document, target, maximum)
    relations = discover_explanations(document)
    moved_notes = {r["text_id"] for r in relations["relations"] if r["inferred"]}
    # Keep a uniquely associated explanation under its visual/table scope,
    # rather than leave a duplicate in an unrelated body-text group.
    for parent in parents:
        if parent.kind == "text":
            parent.parts = [
                p
                for p in parent.parts
                if p.context_only or not moved_notes.intersection(p.source_element_ids)
            ]
    parents = [p for p in parents if any(not item.context_only for item in p.parts)]
    lookup = {e.id: e for e in document.elements}
    budget = TokenBudget()
    all_children = []
    exceptions: set[str] = set()
    for parent in parents:
        parent.id = "semantic-group:" + identity(parent.id, "v2")
        # Normalize array-valued chart series to whole point records before
        # grouping, without splitting JSON through its middle.
        normalized = []
        for item in parent.parts:
            data = json.loads(item.text) if item.role == "chart_series" else {}
            if "points" in data:
                for i, point in enumerate(data["points"]):
                    normalized.append(
                        item.model_copy(
                            update={
                                "id": item.id + f":point:{i}",
                                "text": json.dumps(
                                    dict(
                                        series=data.get("series", data.get("name")),
                                        unit=data.get("unit"),
                                        **point,
                                    ),
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    )
            else:
                normalized.append(item)
        parent.parts = normalized
        anchor_ids = {e.id for e in parent.source_elements if e.type in {"table", "chart", "image"}}
        optional = []
        for relation in relations["relations"]:
            if relation["anchor_id"] not in anchor_ids:
                continue
            note = lookup[relation["text_id"]]
            if not isinstance(note, TextElement):
                continue
            is_caption = relation.get("role") == "caption"
            existing = next(
                (
                    p
                    for p in parent.parts
                    if note.id in p.source_element_ids and p.role in {"text", "footnote", "caption"}
                ),
                None,
            )
            if existing:
                if not is_caption:
                    optional.append(existing.id)
                continue
            item = part(
                note,
                "caption" if is_caption else "footnote" if note.type == "footnote" else "text",
                note.text,
                context_only=is_caption,
                inferred=relation["inferred"],
            )
            item.id += ":related:" + parent.id
            parent.parts.append(item)
            if not is_caption:
                optional.append(item.id)
            if note.id not in {e.id for e in parent.source_elements}:
                parent.source_elements.append(note)
        for ambiguous in relations["ambiguous"]:
            if anchor_ids.intersection(ambiguous["candidate_ids"]):
                parent.quality_warnings.append(
                    "Ambiguous related explanation retained separately: " + ambiguous["text_id"]
                )
        children, oversized = pack_semantic(
            parent, list(dict.fromkeys(optional)), budget, target, maximum, note_limit
        )
        all_children.extend(children)
        exceptions.update(oversized)
    validation = validate_chunks(
        document,
        parents,
        all_children,
        base_validation["excluded_elements"],
        ChunkSettings(child_target_tokens=target, child_max_tokens=maximum),
        budget,
        oversized_child_ids=exceptions,
    )
    validation.update(
        continuation_context_repeated=True,
        semantic_boundaries=True,
        soft_maximum_tokens=maximum,
        oversized_atomic_child_ids=sorted(exceptions),
        related_explanations=len(relations["relations"]),
        ambiguous_explanations=len(relations["ambiguous"]),
    )
    parts_by_parent = {p.id: {part.id: part for part in p.parts} for p in parents}
    for relation in relations["relations"]:
        eid = relation["text_id"]
        relation["inline_child_ids"] = [
            c.id
            for c in all_children
            if any(
                eid in parts_by_parent[c.parent_id][pid].source_element_ids
                for pid in c.context_part_ids
            )
        ]
        relation["standalone_child_ids"] = [
            c.id
            for c in all_children
            if any(
                eid in parts_by_parent[c.parent_id][s.part_id].source_element_ids for s in c.slices
            )
        ]
    return parents, all_children, validation, relations


def build_semantic_bundles(sources: Sequence[Path], destination: Path) -> list[Path]:
    if destination.exists():
        paths = sorted(p.parent for p in destination.glob("*/chunk_manifest.json"))
        if len(paths) != len(sources):
            raise ValueError("Incomplete semantic bundles; preserve and use a new directory")
        for path in paths:
            verify_bundle(path)
        return paths
    paths = []
    for number, source in enumerate(sources, 1):
        verify_bundle(source)
        source_path = source / "source_document.json"
        document = Document.model_validate_json(source_path.read_bytes())
        parents, children, validation, relations = build_semantic_chunks(document)
        output = destination / f"document-{number:02}"
        output.mkdir(parents=True)
        records: list[tuple[str, Sequence[ParentRecord | ChildRecord]]] = [
            ("parents", parents),
            ("children", children),
        ]
        for name, rows in records:
            (output / (name + ".jsonl")).write_text(
                "".join(r.model_dump_json() + "\n" for r in rows)
            )
        save(output / "validation.json", validation)
        save(output / "related_explanations.json", relations)
        save(
            output / "chunk_manifest.json",
            dict(
                document_id=document.document_id,
                algorithm="simple-structure-v2",
                source_document_sha256=digest(source_path),
                target_tokens=350,
                soft_maximum_tokens=500,
                inline_explanation_budget_tokens=100,
                boundaries=(
                    "Whole table rows; whole chart series when possible, otherwise "
                    "whole points; paragraphs/sentences. Indivisible oversized units "
                    "preserved and flagged."
                ),
                context=(
                    "Repeat full heading hierarchy, captions and table headers; related "
                    "explanations only when they fit without displacing primary content."
                ),
                artifacts={
                    name: digest(output / name)
                    for name in [
                        "parents.jsonl",
                        "children.jsonl",
                        "validation.json",
                        "related_explanations.json",
                    ]
                },
            ),
        )
        paths.append(output)
    return paths
