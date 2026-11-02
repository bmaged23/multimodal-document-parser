"""Source coverage, table semantics, provenance and budget invariants."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("tiktoken")

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.rag_preparation.chunk_builder import build_chunks, validate_chunks
from multimodal_doc_parser.rag_preparation.chunk_schema import ChunkSettings
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def fixture_document():
    region = [dict(page_number=1, bounding_box=dict(x0=0.1, y0=0.1, x1=0.9, y1=0.9))]
    provenance = dict(producer="synthetic-test", method="extracted")

    def element(identity, kind, **values):
        return dict(id=identity, type=kind, source_regions=region, provenance=provenance, **values)

    return Document.model_validate(
        dict(
            document_id="sha256:" + "a" * 64,
            source_filename="synthetic.pdf",
            status="parsed",
            preparation=dict(dpi=72, renderer="synthetic"),
            pages=[
                dict(
                    page_number=1,
                    width_points=100,
                    height_points=100,
                    width_pixels=100,
                    height_pixels=100,
                    rotation_degrees=0,
                    image=dict(path="assets/page.png"),
                )
            ],
            elements=[
                element("heading", "heading", text="Forecast", level=1),
                element(
                    "body", "text", parent_id="heading", text="Unicode café 中文 العربية. " * 200
                ),
                element(
                    "table",
                    "table",
                    parent_id="heading",
                    caption_element_id="caption",
                    footnote_element_ids=["note"],
                    row_count=4,
                    column_count=2,
                    logical_table_id="logical-table",
                    column_group="Growth",
                    cells=[
                        dict(row=0, column=0, text="Country", is_header=True, header_inferred=True),
                        dict(
                            row=0, column=1, text="2026 (%)", is_header=True, header_inferred=True
                        ),
                        dict(row=1, column=0, text="Country A", row_span=2),
                        dict(row=1, column=1, text="3.2"),
                        dict(row=2, column=1, text="3.4"),
                        dict(row=3, column=0, text="Country B"),
                        dict(row=3, column=1, text="1.1"),
                    ],
                ),
                element("caption", "caption", text="Table A. Forecast"),
                element("note", "footnote", text="Units are percent. Projections are conditional."),
                element(
                    "image",
                    "image",
                    parent_id="heading",
                    asset=dict(path="assets/photo.png"),
                    description=dict(
                        text="A suspended component.",
                        provenance=dict(producer="gemma4:31b", method="generated"),
                    ),
                ),
                element("footer", "page_footer", text="Page 1"),
            ],
        )
    )


def test_lossless_unicode_slices_stable_ids_and_no_source_mutation():
    document = fixture_document()
    original = document.model_dump_json()
    budget = TokenBudget()
    settings = ChunkSettings(child_target_tokens=90, child_max_tokens=128)
    parents, children, exclusions = build_chunks(document, settings, budget)
    result = validate_chunks(document, parents, children, exclusions, settings, budget)
    assert result["status"] == "passed" and result["max_child_tokens"] <= 128
    assert document.model_dump_json() == original
    again = build_chunks(document, settings, budget)
    assert [c.model_dump() for c in children] == [c.model_dump() for c in again[1]]
    damaged = deepcopy(children)
    damaged[0].slices[0].start += 1
    with pytest.raises(ValueError, match="evidence slices"):
        validate_chunks(document, parents, damaged, exclusions, settings, budget)


def test_table_row_spans_headers_notes_and_generated_provenance():
    parents, children, _ = build_chunks(fixture_document(), ChunkSettings(), TokenBudget())
    table = next(p for p in parents if p.kind == "table")
    rows = [p for p in table.parts if p.role == "table_row"]
    assert len(rows) == 3
    assert "Country A" in rows[0].text and "Country A" in rows[1].text
    assert rows[1].cells[0].row == 1
    assert "2026 (%)" in rows[0].text
    assert any(p.inferred for p in table.parts if p.role == "table_header")
    child = next(c for c in children if c.kind == "table")
    assert "Units are percent" in child.text
    assert child.column_group == "Growth"
    visual = next(p for p in parents if p.kind == "visual")
    assert (
        next(p for p in visual.parts if p.role == "description").provenance[0].method == "generated"
    )


def test_continuations_share_parent_without_mixing_column_groups():
    document = fixture_document()
    first = next(e for e in document.elements if e.type == "table")
    other = first.model_copy(deep=True)
    other.id = "continued-table"
    other.column_group = "Inflation"
    other.cells[1].text = "2026 inflation (%)"
    document.elements.append(other)
    parents, children, excluded = build_chunks(document, ChunkSettings(), TokenBudget())
    validate_chunks(document, parents, children, excluded, ChunkSettings(), TokenBudget())
    tables = [p for p in parents if p.kind == "table"]
    assert len(tables) == 1
    assert {c.column_group for c in children if c.kind == "table"} == {"Growth", "Inflation"}
    for child in (c for c in children if c.kind == "table"):
        assert "Projections are conditional" in child.text
        table_ids = {s.part_id.split(":")[0] for s in child.slices}
        assert len(table_ids) == 1


def test_preview_json_cannot_escape_script_and_bundle_refuses_overwrite(tmp_path):
    from PIL import Image

    from multimodal_doc_parser.rag_preparation.chunk_bundle import prepare_chunks

    document = fixture_document()
    document.elements[1].text = "</script><script>window.injected=true</script>"
    source = tmp_path / "source"
    source.mkdir()
    (source / "assets").mkdir()
    Image.new("RGB", (10, 10)).save(source / "assets/photo.png")
    (source / "document.json").write_text(document.model_dump_json())
    output = prepare_chunks(source, tmp_path / "chunks")
    html = (output / "review.html").read_text()
    assert "</script><script>window.injected" not in html
    assert "\\u003c/script>" in html
    assert json.loads((output / "validation.json").read_text())["status"] == "passed"
    with pytest.raises(ValueError, match="fresh"):
        prepare_chunks(source, output)


def test_keyword_retrieval_expansion_budget_and_corruption_detection(tmp_path):
    from PIL import Image

    from multimodal_doc_parser.rag_preparation.chunk_bundle import prepare_chunks
    from multimodal_doc_parser.rag_preparation.retrieval import Retriever, build_index

    source = tmp_path / "source"
    (source / "assets").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(source / "assets/photo.png")
    (source / "document.json").write_text(fixture_document().model_dump_json())
    bundle = prepare_chunks(source, tmp_path / "chunks")
    index = build_index([bundle], tmp_path / "index")
    retriever = Retriever(index)
    result = retriever.search(
        "Country A 2026 projections", mode="keyword", top_k=2, context_tokens=512
    )
    assert result["matched_children"] and result["contexts"]
    assert result["context_tokens"] <= 512
    assert any("3.2" in c["text"] for c in result["contexts"])
    assert any("Units are percent" in c["text"] for c in result["contexts"])
    assert all(c["source_regions"] for c in result["contexts"])
    with pytest.raises(ValueError, match="requires a GPU"):
        retriever.search("Country A", mode="hybrid")
    (index / "children.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        Retriever(index)
