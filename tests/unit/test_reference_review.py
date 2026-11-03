"""Only explicit source-checked decisions enter references; edits survive reloads."""

import json

import pytest
from PIL import Image

from multimodal_doc_parser.evaluation.reference_review import (
    ReviewStore,
    ReviewUpdate,
    RevisionConflict,
    digest,
    validate_value,
)


@pytest.fixture
def workspace(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (100, 100), "white").save(image)
    dataset = dict(
        pages=[
            dict(
                case="math",
                page=1,
                image="source.png",
                image_sha256=digest(image),
                document_id="sha256:" + "a" * 64,
            )
        ],
        annotations=[
            dict(
                id="a1",
                case="math",
                page=1,
                kind="text",
                title="Source span",
                value=dict(text="written 2+2=5", scope="selected_span"),
                source_box=None,
                provenance="Assistant source draft",
                origin=dict(file="frozen-reference.json"),
                instruction="Check source",
            )
        ],
    )
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    return tmp_path


def update(status="approved", revision=0, box=True):
    return ReviewUpdate(
        expected_revision=revision,
        status=status,
        reviewer="Reviewer",
        value=dict(text="written 2+2=5", scope="selected_span"),
        source_box=dict(x0=0.1, y0=0.1, x1=0.8, y1=0.4) if box else None,
        note="Checked source region",
    )


def test_drafts_do_not_become_ground_truth_without_review(workspace):
    store = ReviewStore(workspace)
    exported = store.export()
    assert exported["annotations"] == [] and exported["counts"]["draft"] == 1
    with pytest.raises(ValueError, match="Draw a source region"):
        store.update("a1", update(box=False))
    with pytest.raises(ValueError, match="reviewer name"):
        store.update("a1", update().model_copy(update=dict(reviewer="")))
    assert store.export()["counts"]["approved"] == 0


def test_source_uncertainty_notes_survive_draft_initialization(workspace):
    path = workspace / "dataset.json"
    dataset = json.loads(path.read_text())
    dataset["annotations"][0]["note"] = "Overwritten bracket; author must verify."
    path.write_text(json.dumps(dataset))
    store = ReviewStore(workspace)
    assert store.snapshot()["annotations"][0]["note"] == dataset["annotations"][0]["note"]
    assert store.snapshot()["annotations"][0]["status"] == "draft"


def test_approved_edits_are_persistent_audited_and_do_not_rewrite_drafts(workspace):
    before = (workspace / "dataset.json").read_bytes()
    store = ReviewStore(workspace)
    value = update().model_copy(
        update=dict(value=dict(text="As actually written", scope="selected_span"))
    )
    store.update("a1", value)
    reloaded = ReviewStore(workspace)
    exported = reloaded.export()
    assert exported["annotations"][0]["value"]["text"] == "As actually written"
    assert exported["annotations"][0]["history"][0]["reviewer"] == "Reviewer"
    assert "not independently annotated" in exported["annotation_status"]
    assert (workspace / "dataset.json").read_bytes() == before
    with pytest.raises(RevisionConflict):
        reloaded.update("a1", update())


def test_ambiguity_is_retained_but_excluded_from_approved_export(workspace):
    store = ReviewStore(workspace)
    store.update("a1", update(status="ambiguous", box=False))
    assert store.export()["annotations"] == []
    assert store.export()["counts"]["ambiguous"] == 1
    with pytest.raises(ValueError, match="Record why"):
        store.update("a1", update(status="ambiguous", revision=1).model_copy(update=dict(note="")))


def test_changed_source_or_draft_identity_cannot_be_silently_approved(workspace):
    store = ReviewStore(workspace)
    (workspace / "source.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        store.update("a1", update())
    d = json.loads((workspace / "dataset.json").read_text())
    d["version"] = "changed"
    # A stored decision is needed to bind a workspace identity across sessions.
    store.state_path.write_text(json.dumps(store.state))
    (workspace / "dataset.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Draft dataset changed"):
        ReviewStore(workspace)


def test_structured_references_reject_duplicate_origins_and_heading_cycles():
    c = dict(row=0, column=0, text="42", row_span=None, column_span=None, is_header=None)
    with pytest.raises(ValueError, match="Duplicate cell"):
        validate_value("table_cells", dict(scope="selected_cells", cells=[c, c]))
    with pytest.raises(ValueError, match="cycle"):
        validate_value(
            "headings", dict(headings=[dict(id="h", text="Heading", level=1, parent_id="h")])
        )
    with pytest.raises(ValueError, match="finite"):
        validate_value(
            "chart_values",
            dict(printed_points=[dict(series="a", label="2024", value=float("nan"))]),
        )
