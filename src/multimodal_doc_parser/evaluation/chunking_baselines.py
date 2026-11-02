"""Controlled flat-chunk baseline over the same saved evidence representation."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from multimodal_doc_parser.evaluation.rag_benchmark import context_evidence_metrics, save
from multimodal_doc_parser.rag_preparation.chunk_bundle import read_records
from multimodal_doc_parser.rag_preparation.chunk_schema import (
    ChildRecord,
    ParentRecord,
    PartSlice,
)
from multimodal_doc_parser.rag_preparation.retrieval import Retriever
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def fixed_spans(
    text: str, budget: TokenBudget, size: int = 350, overlap: int = 50
) -> Iterator[tuple[int, int]]:
    """Character-safe fixed token windows; no semantic boundary preference."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    start = 0
    while start < len(text):
        low, high, end = start + 1, len(text), start
        while low <= high:
            middle = (low + high) // 2
            if budget.count(text[start:middle]) <= size:
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError("A character exceeds the chunk budget")
        yield start, end
        if end == len(text):
            break
        low, high, next_start = start + 1, end, end
        while low <= high:
            middle = (low + high) // 2
            if budget.count(text[middle:end]) <= overlap:
                next_start, high = middle, middle - 1
            else:
                low = middle + 1
        start = max(start + 1, next_start)


def build_fixed_bundles(index: Path, destination: Path) -> list[Path]:
    """Flatten saved parent parts, retaining exact part-offset provenance.

    This deliberately holds the table/chart textual representation constant. It
    tests chunk boundaries and expansion, not a different parser or serializer.
    Context-only parts are kept once per original parent; no new context is
    injected at query time. Original parent order is retained and recorded.
    """
    if destination.exists():
        bundles = sorted(p.parent for p in destination.glob("*/chunk_manifest.json"))
        if len(bundles) != 3:
            raise ValueError("Incomplete baseline bundles; preserve them and use a fresh directory")
        return bundles
    grouped: dict[str, list[ParentRecord]] = defaultdict(list)
    for parent in read_records(index / "parents.jsonl", ParentRecord):
        grouped[parent.document_id].append(parent)
    budget = TokenBudget()
    bundles = []
    for number, (document_id, originals) in enumerate(grouped.items(), 1):
        directory = destination / f"document-{number:02}"
        directory.mkdir(parents=True)
        parts = []
        elements = {}
        intervals = []
        text = ""
        for parent in originals:
            elements.update({e.id: e for e in parent.source_elements})
            for part in parent.parts:
                # Some source parts occur in multiple parents with the same ID.
                copied = part.model_copy(update={"id": parent.id + "/" + part.id})
                parts.append(copied)
                method = ",".join(sorted({p.method for p in part.provenance}))
                flags = "; inferred structure" if part.inferred else ""
                label = f"[{part.role}; {method}{flags}]\n"
                text += label
                begin = len(text)
                text += part.text
                intervals.append((begin, len(text), copied.id))
                text += "\n\n"
        parent = ParentRecord(
            id="fixed-parent:" + hashlib.sha256(document_id.encode()).hexdigest()[:24],
            document_id=document_id,
            section_id=None,
            heading_path=[],
            kind="text",
            source_elements=list(elements.values()),
            parts=parts,
        )
        children = []
        for i, (start, end) in enumerate(fixed_spans(text, budget)):
            slices = [
                PartSlice(part_id=pid, start=max(start, a) - a, end=min(end, b) - a)
                for a, b, pid in intervals
                if max(start, a) < min(end, b)
            ]
            if not slices:
                raise ValueError("Window contains no evidence text")
            lookup = {p.id: p for p in parts}
            source_ids = list(
                dict.fromkeys(e for s in slices for e in lookup[s.part_id].source_element_ids)
            )
            children.append(
                ChildRecord(
                    id=f"fixed-child:{number:02}:{i:05}",
                    parent_id=parent.id,
                    document_id=document_id,
                    heading_path=[],
                    kind="text",
                    text=text[start:end],
                    token_count=budget.count(text[start:end]),
                    slices=slices,
                    context_part_ids=[],
                    deferred_context_part_ids=[],
                    source_element_ids=source_ids,
                    source_regions=[r for eid in source_ids for r in elements[eid].source_regions],
                )
            )
        parent.child_ids = [c.id for c in children]
        (directory / "parents.jsonl").write_text(parent.model_dump_json() + "\n")
        (directory / "children.jsonl").write_text(
            "".join(c.model_dump_json() + "\n" for c in children)
        )
        save(
            directory / "chunk_manifest.json",
            dict(
                document_id=document_id,
                algorithm="fixed-350-overlap-50",
                representation="Same saved evidence parts; original parent/part order",
                artifacts={
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ["parents.jsonl", "children.jsonl"]
                },
            ),
        )
        bundles.append(directory)
    return bundles


class FixedRetriever(Retriever):
    """Supply only fixed windows; never silently inject adaptive parent context."""

    def expand(self, matched: list[str], maximum: int, expand: bool) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for identity in matched:
            child = self.by_id[identity]
            prefix = f"Evidence chunk {child.id}; document {child.document_id}\n"
            text = prefix + child.text
            if self.budget.count("\n\n".join([b["text"] for b in output] + [text])) > maximum:
                continue
            output.append(
                dict(
                    parent_id=child.parent_id,
                    document_id=child.document_id,
                    matched_child_ids=[child.id],
                    source_element_ids=child.source_element_ids,
                    source_regions=[r.model_dump() for r in child.source_regions],
                    evidence_slices=[s.model_dump() for s in child.slices],
                    context_part_ids=[],
                    deferred_context_part_ids=[],
                    quality_warnings=[],
                    text=text,
                )
            )
        return output


def merged_source_coverage(
    question: Any, contexts: list[dict[str, Any]], parents: dict[str, Any]
) -> dict[str, Any]:
    """Credit complete parts covered by the union of retrieved fragments.

    Splitting a row across windows is allowed, but an isolated row fragment or
    mere table-ID hit cannot earn full-row evidence credit.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in contexts:
        groups[block["parent_id"]].append(block)
    merged = []
    for pid, blocks in groups.items():
        parent = parents[pid]
        spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        contextual = set()
        for block in blocks:
            contextual.update(block["context_part_ids"])
            for s in block["evidence_slices"]:
                spans[s["part_id"]].append((s["start"], s["end"]))
        selected = []
        for part in parent.parts:
            end = 0
            for a, b in sorted(spans[part.id]):
                if a > end:
                    break
                end = max(end, b)
            if end >= len(part.text) and part.text:
                selected.append(dict(part_id=part.id, start=0, end=len(part.text)))
        merged.append(
            dict(
                parent_id=pid,
                document_id=parent.document_id,
                evidence_slices=selected,
                context_part_ids=list(contextual),
            )
        )
    return context_evidence_metrics(question, {"contexts": merged}, parents)


def cell_aware_source_coverage(
    question: Any, contexts: list[dict[str, Any]], parents: dict[str, Any]
) -> dict[str, Any]:
    """Measure selected cells rather than penalizing every split of unrelated columns.

    A table cell requires its complete serialized column-label/value segment.
    Header parts require complete text. Narrative paragraphs and chart points
    retain the conservative complete-part rule. All methods use the same rule.
    """
    if question.expected_abstention:
        return dict(applicable=False)
    spans: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for block in contexts:
        parent = parents[block["parent_id"]]
        for s in block["evidence_slices"]:
            spans[(parent.id, s["part_id"])].append((s["start"], s["end"]))
        for pid in block["context_part_ids"]:
            part = next(p for p in parent.parts if p.id == pid)
            spans[(parent.id, pid)].append((0, len(part.text)))

    def covered(pid: str, part_id: str, begin: int, end: int) -> bool:
        cursor = begin
        for a, b in sorted(spans[(pid, part_id)]):
            if b <= cursor:
                continue
            if a > cursor:
                return False
            cursor = b
            if cursor >= end:
                return True
        return False

    flags = []
    for need in question.evidence:
        found = False
        sources = [
            dict(document_id=need.document_id, element_id=need.element_id)
        ] + need.alternative_sources
        for source in sources:
            cells: set[tuple[int, int]] = set()
            labels = set()
            text_found = False
            for parent in parents.values():
                if parent.document_id != source["document_id"]:
                    continue
                for part in parent.parts:
                    if source["element_id"] not in part.source_element_ids:
                        continue
                    if covered(parent.id, part.id, 0, len(part.text)):
                        cells.update(
                            (c.row, c.column)
                            for c in part.cells
                            if c.element_id == source["element_id"]
                        )
                        if part.role == "chart_series":
                            data = json.loads(part.text)
                            if data.get("series") == need.series:
                                labels.add(data.get("label"))
                        if part.role not in ["heading", "caption", "table_header"]:
                            text_found = True
                    elif part.role == "table_row":
                        # The serializer emits one segment per CellReference, in order.
                        prefix, separator, body = part.text.partition(": ")
                        segments = body.split(" | ") if separator else []
                        if len(segments) != len(part.cells):
                            continue  # Ambiguous embedded delimiters never earn guessed credit.
                        offset = len(prefix) + len(separator)
                        for segment, cell in zip(segments, part.cells, strict=True):
                            if cell.element_id == source["element_id"] and covered(
                                parent.id, part.id, offset, offset + len(segment)
                            ):
                                cells.add((cell.row, cell.column))
                            offset += len(segment) + 3
            found = (
                set(need.cells).issubset(cells)
                if need.cells
                else set(need.point_labels).issubset(labels)
                if need.series
                else text_found
            )
            if found:
                break
        flags.append(found)
    return dict(
        applicable=True, recall=sum(flags) / len(flags), complete=all(flags), found=sum(flags)
    )
