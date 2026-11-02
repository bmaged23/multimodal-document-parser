"""Public excerpts retain source mappings/failures and exclude unselected material."""

import json

import pypdfium2 as pdfium
import pytest

from multimodal_doc_parser.evaluation.public_demo import build_public_demo
from multimodal_doc_parser.evaluation.showcase import build_showcase


def selection_file(project, pages):
    path = project / "selection.json"
    path.write_text(
        json.dumps(
            dict(
                version="1.0",
                cases=[
                    dict(
                        id="fixture-public",
                        saved_pages=pages,
                        attribution="Synthetic fixture, not a model result.",
                        rights_note="Project-created test content.",
                    )
                ],
            )
        )
    )
    return path


def test_public_excerpt_mapping_failures_and_asset_exclusion(showcase_project):
    project, manifest = showcase_project
    source = project / "showcase"
    build_showcase(project, source, manifest)
    destination = project / "public-demo"
    build_public_demo(source, destination, selection_file(project, [2]))
    catalog = json.loads((destination / "catalog.json").read_text())
    case = catalog["cases"][0]
    assert case["original_pages"] == [20]
    assert case["page_count"] == 1
    original_case = json.loads((source / "catalog.json").read_text())["cases"][0]
    assert case["description"].startswith(original_case["description"])
    assert "Synthetic fixture, not a model result." in case["description"]
    with pdfium.PdfDocument(destination / case["source"]) as excerpt:
        assert len(excerpt) == 1
        page = excerpt[0]
        text = page.get_textpage()
        assert "Synthetic source page 2" in text.get_text_range()
        text.close()
        page.close()
    payload = json.loads((destination / "runs/model-a/run-data.json").read_text())
    assert len(payload["pages"]) == 1
    assert payload["pages"][0]["number"] == 1
    assert payload["pages"][0]["saved_page_number"] == 2
    assert payload["pages"][0]["status"] == "not_accepted"
    assert payload["metrics"]["completed"] == 1
    assert "full saved selection" in payload["metrics"]["timing_scope"]
    assert not (destination / "runs/model-a/pages/0001-content.html").exists()
    assert not (destination / "runs/model-a/assets/pages/page_0001.png").exists()
    assert not (destination / "runs/model-a/document.json").exists()
    assert not (destination / "runs/private-a").exists()
    assert json.loads((destination / "build-manifest.json").read_text())["inference_requests"] == 0


def test_public_demo_refuses_private_showcases_and_checksum_changes(showcase_project):
    project, manifest = showcase_project
    private = project / "private-showcase"
    build_showcase(project, private, manifest, include_private=True)
    selection = selection_file(project, [1])
    with pytest.raises(ValueError, match="private inputs excluded"):
        build_public_demo(private, project / "private-demo", selection)
    public = project / "showcase"
    build_showcase(project, public, manifest)
    (public / "runs/model-a/pages/0001-content.html").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_public_demo(public, project / "tampered-demo", selection)
    assert not (project / "tampered-demo").exists()
    assert not list(project.glob(".public-demo-*"))


@pytest.mark.parametrize("pages", [[0], [3], [1, 1]])
def test_public_demo_rejects_invalid_page_selections(showcase_project, pages):
    project, manifest = showcase_project
    source = project / "showcase"
    build_showcase(project, source, manifest)
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        build_public_demo(source, project / "invalid-demo", selection_file(project, pages))
