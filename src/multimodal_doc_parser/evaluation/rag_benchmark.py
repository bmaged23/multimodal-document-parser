"""Source-addressable benchmark contracts and deterministic evidence metrics."""

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from multimodal_doc_parser.document_schema.document import SchemaModel

CATEGORY_COUNTS = dict(narrative=20, table=30, chart=15, image=5, cross_page=20, unanswerable=10)


class EvidenceNeed(SchemaModel):
    document_id: str
    element_id: str
    pages: list[int]
    cells: list[tuple[int, int]] = Field(default_factory=list)
    series: str | None = None
    point_labels: list[str] = Field(default_factory=list)
    accepted_child_ids: list[str] = Field(min_length=1)
    accepted_parent_ids: list[str] = Field(min_length=1)
    source_text: str
    alternative_sources: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkQuestion(SchemaModel):
    id: str = Field(pattern=r"^q\d{3}$")
    category: Literal["narrative", "table", "chart", "image", "cross_page", "unanswerable"]
    question: str = Field(min_length=15)
    reference: str = Field(min_length=5)
    required_facts: list[str]
    expected_abstention: bool
    evidence: list[EvidenceNeed]
    group_id: str
    split: Literal["development", "test"]
    reference_status: str
    numeric_values: list[str] = Field(default_factory=list)
    reference_draft: str | None = None
    reference_generator_model: str | None = None
    derivation: dict[str, Any] | None = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_dataset(questions: list[BenchmarkQuestion]) -> dict[str, Any]:
    if len(questions) != 100 or len({q.id for q in questions}) != 100:
        raise ValueError("Benchmark requires exactly 100 unique question IDs")
    if Counter(q.category for q in questions) != Counter(CATEGORY_COUNTS):
        raise ValueError("Question category allocation differs from the agreed 100-question plan")
    splits = Counter(q.split for q in questions)
    if splits not in [Counter(development=30, test=70), Counter(test=100)]:
        raise ValueError("Require a grouped 30/70 split or a frozen all-100 test benchmark")
    groups: dict[str, set[str]] = {}
    for question in questions:
        groups.setdefault(question.group_id, set()).add(question.split)
        if question.expected_abstention != (question.category == "unanswerable"):
            raise ValueError("Abstention label does not match question category")
        if not question.expected_abstention and not question.evidence:
            raise ValueError("Answerable question requires saved supporting evidence")
        if (
            question.category == "cross_page"
            and len({(e.document_id, p) for e in question.evidence for p in e.pages}) < 2
        ):
            raise ValueError("Cross-page question lacks evidence on multiple pages")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Related evidence group leaked across development/test splits")
    return dict(
        status="passed",
        questions=100,
        categories=dict(CATEGORY_COUNTS),
        splits=dict(splits),
        groups=len(groups),
    )


def retrieval_metrics(
    question: BenchmarkQuestion, result: dict[str, Any], *, k: int = 5
) -> dict[str, Any]:
    """Recall over required evidence groups with alternate valid child IDs."""
    if question.expected_abstention:
        return dict(
            applicable=False, reason="No positive gold evidence exists for an unanswerable question"
        )
    ranked = [m["child_id"] for m in result["matched_children"]]
    flags = [any(c in need.accepted_child_ids for c in ranked[:k]) for need in question.evidence]
    reciprocal = next(
        (
            1 / i
            for i, c in enumerate(ranked[:k], 1)
            if any(c in e.accepted_child_ids for e in question.evidence)
        ),
        0.0,
    )
    # Context coverage is anchored to source/cell evidence, not merely a parent
    # ID: a huge parent hit must not count as retrieving every row in it.
    return dict(
        applicable=True,
        k=k,
        required_evidence_groups=len(flags),
        evidence_groups_found=sum(flags),
        evidence_recall=sum(flags) / len(flags),
        complete_evidence_success=all(flags),
        reciprocal_rank=reciprocal,
    )


def numeric_checks(question: BenchmarkQuestion, answer: str) -> dict[str, Any]:
    """Exact expected numeric mentions only; not a numerical association score."""
    normalized = re.sub(r"(?<=\d)[,\s](?=\d{3}(?:\D|$))", "", answer.replace("−", "-"))
    tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:/\d+)?", normalized)
    found = []
    for value in question.numeric_values:
        expected = value.replace(" ", "").replace(",", "")
        if "/" in expected:
            found.append(expected in tokens)
        else:
            found.append(
                any("/" not in token and Decimal(token) == Decimal(expected) for token in tokens)
            )
    return dict(
        applicable=bool(found),
        expected=question.numeric_values,
        found=found,
        all_expected_mentions=all(found) if found else None,
        scope=(
            "Checks mentions, not whether numbers are assigned to the correct quantities or units"
        ),
    )


def context_evidence_metrics(
    question: BenchmarkQuestion, result: dict[str, Any], parents: dict[str, Any]
) -> dict[str, Any]:
    if question.expected_abstention:
        return dict(applicable=False)
    flags = []
    for need in question.evidence:
        found = False
        sources = [
            dict(document_id=need.document_id, element_id=need.element_id)
        ] + need.alternative_sources
        for block in result.get("contexts", []):
            parent = parents.get(block.get("parent_id"))
            if parent is None:
                continue
            selected = {
                piece["part_id"]
                for piece in block["evidence_slices"]
                if piece["start"] == 0
                and piece["end"]
                == len(next(p.text for p in parent.parts if p.id == piece["part_id"]))
            }
            selected.update(block["context_part_ids"])
            for source in sources:
                if source["document_id"] != block["document_id"]:
                    continue
                parts = [
                    p
                    for p in parent.parts
                    if p.id in selected and source["element_id"] in p.source_element_ids
                ]
                if not parts:
                    continue
                if need.cells:
                    cells = {
                        (c.row, c.column)
                        for p in parts
                        for c in p.cells
                        if c.element_id == source["element_id"]
                    }
                    element = next(
                        e for e in parent.source_elements if e.id == source["element_id"]
                    )
                    for cell in getattr(element, "cells", []):
                        if (
                            cell.is_header
                            and cell.text
                            and any(p.role == "table_row" and cell.text in p.text for p in parts)
                        ):
                            cells.add((cell.row, cell.column))
                    found = set(need.cells).issubset(cells)
                elif need.series:
                    labels = set()
                    for p in parts:
                        if p.role == "chart_series":
                            series = json.loads(p.text)
                            if series.get("series") == need.series:
                                labels.add(series.get("label"))
                    found = set(need.point_labels).issubset(labels)
                else:
                    found = any(p.role not in ["heading", "caption", "table_header"] for p in parts)
                if found:
                    break
            if found:
                break
        flags.append(found)
    return dict(
        applicable=True,
        required_evidence_groups=len(flags),
        found=sum(flags),
        recall=sum(flags) / len(flags),
        complete=all(flags),
    )


def unit_checks(question: BenchmarkQuestion, answer: str) -> dict[str, Any]:
    reference = question.reference.casefold()
    text = answer.casefold()
    expected = []
    checks = []
    units = [
        (
            "USD/kWh",
            "usd/kwh" in reference,
            bool(re.search(r"(?:/\s*kwh|per\s+(?:kwh|kilowatt[- ]?hour))", text))
            and bool(re.search(r"usd|\$|dollar", text)),
        ),
        (
            "USD/kW",
            bool(re.search(r"usd/kw\b", reference)),
            bool(re.search(r"(?:/\s*kw\b|per\s+(?:kw\b|kilowatt\b))", text))
            and bool(re.search(r"usd|\$|dollar", text)),
        ),
        ("GW", bool(re.search(r"\bgw\b", reference)), bool(re.search(r"\bgw\b|gigawatt", text))),
        ("ms", bool(re.search(r"\bms\b", reference)), bool(re.search(r"\bms\b|millisecond", text))),
        ("percent", "%" in reference, bool(re.search(r"%|percent|per cent", text))),
        (
            "USD billion",
            bool(re.search(r"usd.*?billion", reference)),
            bool(re.search(r"billion|\bbn\b", text)) and bool(re.search(r"usd|\$|dollar", text)),
        ),
    ]
    for name, applicable, present in units:
        if applicable:
            expected.append(name)
            checks.append(present)
    return dict(
        applicable=bool(expected),
        expected=expected,
        present=checks,
        all_present=all(checks) if checks else None,
        scope=(
            "Unit mentions with documented synonyms; quantity/value association is "
            "assessed by the judge, not established by this lexical check."
        ),
    )
