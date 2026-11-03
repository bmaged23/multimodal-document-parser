"""Build source-matched showcase bundles without inferring missing evidence."""

import json
from pathlib import Path

import pytest

from multimodal_doc_parser.evaluation.showcase import build_showcase, safe_file


def test_showcase_preserves_partial_status_raw_separation_and_private_default(showcase_project):
    project, manifest = showcase_project
    output = project / "showcase"
    build_showcase(project, output, manifest)
    catalog = json.loads((output / "catalog.json").read_text())
    assert [c["id"] for c in catalog["cases"]] == ["fixture-public"]
    assert not catalog["private_inputs_included"]
    assert not catalog["runs"]["missing-model"]["available"]
    data = json.loads((output / "runs/model-a/run-data.json").read_text())
    assert data["metrics"]["completed"] == 1
    assert data["pages"][1]["status"] == "not_accepted"
    assert "Saved synthetic raw block" in data["pages"][0]["raw"]
    assert "do-not-copy" not in data["pages"][0]["raw"]
    page = (output / "runs/model-a/pages/0001-preview.html").read_text()
    assert "<script>window.injected=true</script>" not in page
    assert "Content-Security-Policy" in page
    assert json.loads((output / "build-manifest.json").read_text())["inference_requests"] == 0
    with pytest.raises(ValueError, match="fresh showcase"):
        build_showcase(project, output, manifest)


def test_showcase_rejects_different_source_identities_atomically(showcase_project):
    project, manifest = showcase_project
    source = project / "outputs/parser_runs/run-b/document.json"
    data = json.loads(source.read_text())
    data["document_id"] = "sha256:" + "0" * 64
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source identity"):
        build_showcase(project, project / "showcase", manifest)
    assert not (project / "showcase").exists()
    assert not list(project.glob(".showcase-*"))


def test_missing_inference_metadata_is_not_reported_as_success(showcase_project):
    project, manifest = showcase_project
    (project / "outputs/parser_runs/run-a/raw/paddleocr/inference-manifest.json").unlink()
    build_showcase(project, project / "showcase", manifest, include_private=True)
    data = json.loads((project / "showcase/runs/model-a/run-data.json").read_text())
    assert data["metrics"]["completed"] is None
    assert data["pages"][0]["status"] == "unknown"
    assert len(json.loads((project / "showcase/catalog.json").read_text())["cases"]) == 2


def test_asset_paths_cannot_escape_the_selected_run(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not expose")
    (root / "alias.txt").symlink_to(secret)
    for name in ["../secret.txt", "alias.txt"]:
        with pytest.raises(ValueError, match="unsafe"):
            safe_file(root, name)


def test_gemma_description_is_separate_from_the_ocr_card(showcase_project):
    project, manifest = showcase_project
    run = project / "outputs/parser_runs/run-a"
    path = run / "document.json"
    document = json.loads(path.read_text())
    document["elements"].append(
        {
            "id": "visual-one",
            "type": "image",
            "source_regions": [
                {"page_number": 1, "bounding_box": {"x0": 0.1, "y0": 0.4, "x1": 0.9, "y1": 0.9}}
            ],
            "asset": document["pages"][0]["image"],
            "provenance": {"producer": "synthetic OCR", "method": "extracted"},
            "description": {
                "text": "Generated description <script>bad()</script>",
                "provenance": {
                    "producer": "Ollama Cloud",
                    "model_version": "gemma4:31b",
                    "method": "generated",
                },
            },
        }
    )
    path.write_text(json.dumps(document))
    output = project / "gemma-showcase"
    build_showcase(project, output, manifest)
    html = (output / "runs/model-a/pages/0001-content.html").read_text()
    assert 'data-producer="ocr"' in html
    assert 'data-producer="generated"' in html
    assert "OCR · model-a" in html
    assert "Gemma · gemma4:31b" in html
    assert "<script>bad()</script>" not in html
    assert html.count("Generated description") == 1


def test_failed_normalization_still_shows_the_saved_raw_content(showcase_project):
    project, manifest = showcase_project
    run = project / "outputs/parser_runs/run-a"
    (run / "raw/paddleocr/page_0002.json").write_text(
        json.dumps({"content": "Readable model text <script>bad()</script>"})
    )
    output = project / "raw-fallback-showcase"
    build_showcase(project, output, manifest)
    html = (output / "runs/model-a/pages/0002-content.html").read_text()
    assert "No accepted extraction" in html
    assert "Saved raw OCR reply" in html
    assert "Readable model text" in html
    assert "<script>bad()</script>" not in html
