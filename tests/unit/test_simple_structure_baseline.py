"""Practical baseline retains heading scope, table headers and source evidence."""

from copy import deepcopy

from test_adaptive_chunks import fixture_document

from multimodal_doc_parser.evaluation.simple_structure_baseline import build_simple_chunks
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def test_every_body_continuation_keeps_its_full_heading_hierarchy():
    document = fixture_document()
    parent = document.elements[0]
    nested = parent.model_copy(
        update={
            "id": "subheading",
            "parent_id": parent.id,
            "text": "Battery measurements",
            "level": 2,
        }
    )
    body = document.elements[1]
    body.parent_id = nested.id
    document.elements.insert(1, nested)
    before = document.model_dump_json()
    parents, children, validation = build_simple_chunks(document, target=180, maximum=300)
    text_parent = next(p for p in parents if any(s.id == body.id for s in p.source_elements))
    chunks = [c for c in children if c.parent_id == text_parent.id]
    assert len(chunks) > 2
    assert all(c.heading_path == ["Forecast", "Battery measurements"] for c in chunks)
    assert all("Forecast" in c.text and "Battery measurements" in c.text for c in chunks)
    assert all(c.token_count <= 300 for c in children)
    assert validation["continuation_context_repeated"]
    assert document.model_dump_json() == before


def test_table_and_visual_chunks_stay_separate_and_repeat_table_headers():
    document = fixture_document()
    table = next(e for e in document.elements if e.type == "table")
    template = deepcopy(table.cells[-2:])
    for row in range(4, 35):
        for cell in template:
            copy = cell.model_copy(update={"row": row})
            table.cells.append(copy)
    table.row_count = 35
    parents, children, _ = build_simple_chunks(document, target=150, maximum=300)
    table_parent = next(p for p in parents if p.kind == "table")
    table_chunks = [c for c in children if c.parent_id == table_parent.id]
    assert len(table_chunks) > 2
    assert all("Table A. Forecast" in c.text and "2026 (%)" in c.text for c in table_chunks)
    assert all("Country" in c.text and "Forecast" in c.text for c in table_chunks)
    assert all(not ({"body", "image"} & set(c.source_element_ids)) for c in table_chunks)
    visual = [c for c in children if c.kind == "visual"]
    assert visual and all("A suspended component" in c.text for c in visual)
    assert all(not ({"body", "table"} & set(c.source_element_ids)) for c in visual)


def test_empty_header_stub_retains_real_column_names_without_mutating_source():
    document = fixture_document()
    table = next(e for e in document.elements if e.type == "table")
    table.cells[0].text = ""
    table.cells[0].is_header = False
    before = document.model_dump_json()
    parents, children, _ = build_simple_chunks(document)
    header = next(p for parent in parents for p in parent.parts if p.role == "table_header")
    assert "2026 (%)" in header.text
    assert header.inferred
    assert all(TokenBudget().count(c.text) <= 500 for c in children)
    assert document.model_dump_json() == before
