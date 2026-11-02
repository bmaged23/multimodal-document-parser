"""Controlled chunk comparisons must preserve text and source evidence coverage."""

from types import SimpleNamespace

import pytest

from multimodal_doc_parser.evaluation.chunking_baselines import (
    cell_aware_source_coverage,
    fixed_spans,
    merged_source_coverage,
)
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def test_fixed_windows_are_bounded_overlapping_and_lossless():
    budget = TokenBudget()
    text = "Solar panels ☀️ and turbines. جدول بيانات 中文.\n" * 100
    spans = list(fixed_spans(text, budget, size=64, overlap=12))
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(budget.count(text[a:b]) <= 64 for a, b in spans)
    assert all(
        a < next_a < b <= next_b for (a, b), (next_a, next_b) in zip(spans, spans[1:], strict=False)
    )
    assert set(range(len(text))) == {i for a, b in spans for i in range(a, b)}
    with pytest.raises(ValueError):
        list(fixed_spans(text, budget, size=10, overlap=10))


def test_source_coverage_merges_fragments_without_crediting_a_missing_gap():
    part = SimpleNamespace(
        id="part", text="abcdefghij", source_element_ids=["element"], role="text", cells=[]
    )
    parents = {"parent": SimpleNamespace(parts=[part], document_id="doc")}
    need = SimpleNamespace(
        document_id="doc", element_id="element", alternative_sources=[], cells=[], series=None
    )
    question = SimpleNamespace(expected_abstention=False, evidence=[need])

    def block(start, end):
        return dict(
            parent_id="parent",
            document_id="doc",
            context_part_ids=[],
            evidence_slices=[dict(part_id="part", start=start, end=end)],
        )

    assert merged_source_coverage(question, [block(0, 5)], parents)["recall"] == 0
    assert merged_source_coverage(question, [block(0, 5), block(6, 10)], parents)["recall"] == 0
    assert merged_source_coverage(question, [block(0, 5), block(5, 10)], parents)["recall"] == 1


def test_cell_coverage_does_not_require_unrequested_columns_or_accept_wrong_cells():
    text = "Source row 2: Sensor: ZX-19 | Latency: 18.4 | Error: 1.8"
    cells = [SimpleNamespace(row=1, column=i, element_id="table") for i in range(3)]
    part = SimpleNamespace(
        id="row", text=text, source_element_ids=["table"], role="table_row", cells=cells
    )
    parents = {"parent": SimpleNamespace(id="parent", parts=[part], document_id="doc")}
    need = SimpleNamespace(
        document_id="doc",
        element_id="table",
        alternative_sources=[],
        cells=[(1, 0), (1, 1)],
        series=None,
    )
    question = SimpleNamespace(expected_abstention=False, evidence=[need])

    def block(a, b):
        return dict(
            parent_id="parent",
            document_id="doc",
            context_part_ids=[],
            evidence_slices=[dict(part_id="row", start=a, end=b)],
        )

    end = text.index(" | Error")
    # All requested cells are present even though the unrelated Error cell is absent.
    assert cell_aware_source_coverage(question, [block(0, end)], parents)["recall"] == 1
    assert cell_aware_source_coverage(question, [block(0, end - 1)], parents)["recall"] == 0
    assert (
        cell_aware_source_coverage(question, [block(text.index("Error"), len(text))], parents)[
            "recall"
        ]
        == 0
    )
