"""Respect semantic boundaries and preserve bounded, source-linked explanations."""

import json

from test_adaptive_chunks import fixture_document

from multimodal_doc_parser.document_schema.document import ChartElement, TextElement
from multimodal_doc_parser.evaluation.related_explanations import discover_explanations
from multimodal_doc_parser.evaluation.semantic_structure_baseline import build_semantic_chunks


def add_below_note(document, text):
    image = next(e for e in document.elements if e.type == "image")
    image.caption = "Figure 1. Lift arrangement"
    image.source_regions[0].bounding_box.y1 = 0.5
    note = TextElement.model_validate(
        dict(
            id="explanation",
            type="text",
            parent_id=image.parent_id,
            text=text,
            provenance=dict(producer="synthetic-test", method="extracted"),
            source_regions=[
                dict(page_number=1, bounding_box=dict(x0=0.1, y0=0.51, x1=0.9, y1=0.6))
            ],
        )
    )
    document.elements.insert(document.elements.index(image) + 1, note)
    return image, note


def test_related_below_explanation_is_kept_whole_with_the_visual_when_it_fits():
    document = fixture_document()
    image, note = add_below_note(
        document, "Note: this component is suspended; motion is not established."
    )
    before = document.model_dump_json()
    parents, children, validation, relations = build_semantic_chunks(document)
    related = [
        c for c in children if image.id in c.source_element_ids and note.id in c.source_element_ids
    ]
    assert related and any(note.text in c.text for c in related)
    assert all("Forecast" in c.text and image.caption in c.text for c in related)
    assert all(c.token_count <= 500 for c in children)
    assert any(
        r["anchor_id"] == image.id and r["text_id"] == note.id for r in relations["relations"]
    )
    assert validation["semantic_boundaries"]
    assert document.model_dump_json() == before


def test_long_explanation_is_linked_separately_not_truncated_to_fit():
    document = fixture_document()
    image, note = add_below_note(document, "Note: " + "suspended " * 800 + "component.")
    parents, children, validation, relations = build_semantic_chunks(document)
    note_children = [c for c in children if note.text in c.text]
    assert len(note_children) == 1
    assert note_children[0].token_count > 500
    assert note_children[0].id in validation["oversized_atomic_child_ids"]
    assert "Figure 1. Lift arrangement" in note_children[0].text
    assert "Forecast" in note_children[0].text
    relation = next(r for r in relations["relations"] if r["text_id"] == note.id)
    assert relation["standalone_child_ids"]
    assert not relation["inline_child_ids"]


def test_oversized_table_row_is_preserved_as_one_atomic_unit():
    document = fixture_document()
    table = next(e for e in document.elements if e.type == "table")
    table.cells[3].text = "measurement " * 700
    parents, children, validation, _ = build_semantic_chunks(document)
    assert validation["oversized_atomic_child_ids"]
    parts = {p.id: p for parent in parents for p in parent.parts}
    for child in children:
        for piece in child.slices:
            if parts[piece.part_id].role == "table_row":
                assert piece.start == 0 and piece.end == len(parts[piece.part_id].text)


def test_side_by_side_visuals_do_not_claim_the_same_ambiguous_below_paragraph():
    document = fixture_document()
    image, note = add_below_note(document, "Note: these values are approximate.")
    other = image.model_copy(deep=True, update={"id": "second-image"})
    document.elements.insert(document.elements.index(note), other)
    relations = discover_explanations(document)
    assert not any(r["text_id"] == note.id for r in relations["relations"])
    assert any(r["text_id"] == note.id for r in relations["ambiguous"])


def test_adjacent_source_caption_repeats_in_every_visual_continuation():
    document = fixture_document()
    image, note = add_below_note(document, "Note: measurements are approximate.")
    image.caption = None
    image.source_regions[0].bounding_box.y0 = 0.2
    image.description.text = "A component is visible. " * 200
    caption = TextElement.model_validate(
        dict(
            id="source-caption",
            type="caption",
            parent_id=image.parent_id,
            text="Figure 2. Component outline",
            provenance=dict(producer="test", method="extracted"),
            source_regions=[
                dict(page_number=1, bounding_box=dict(x0=0.1, y0=0.11, x1=0.9, y1=0.19))
            ],
        )
    )
    document.elements.insert(document.elements.index(image), caption)
    _, children, _, relations = build_semantic_chunks(document)
    visual = [c for c in children if c.kind == "visual"]
    assert len(visual) > 1
    assert all(caption.text in c.text and "Forecast" in c.text for c in visual)
    assert any(
        r.get("role") == "caption" and r["text_id"] == caption.id for r in relations["relations"]
    )


def test_oversized_chart_splits_between_complete_point_records():
    document = fixture_document()
    image = next(e for e in document.elements if e.type == "image")
    chart = ChartElement.model_validate(
        dict(
            **{**image.model_dump(), "type": "chart"},
            series=[
                dict(
                    name=name,
                    unit="USD/kWh",
                    points=[
                        dict(label=str(2000 + i), value=i / 100, value_basis="printed_label")
                        for i in range(20)
                    ],
                )
                for name in ["Solar", "Wind"]
            ],
        )
    )
    document.elements[document.elements.index(image)] = chart
    parents, children, _, _ = build_semantic_chunks(document)
    parts = {p.id: p for parent in parents for p in parent.parts}
    visual = [c for c in children if c.kind == "visual"]
    assert len(visual) > 1
    points = []
    for child in visual:
        for piece in child.slices:
            item = parts[piece.part_id]
            if item.role == "chart_series":
                assert piece.start == 0 and piece.end == len(item.text)
                points.append(json.loads(item.text))
    assert len(points) == 40
    assert {p["series"] for p in points} == {"Solar", "Wind"}
