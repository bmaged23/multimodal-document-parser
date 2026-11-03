"""References must be restricted to the exact public source-page allowlist."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reference_viewer",
    Path(__file__).resolve().parents[2] / "scripts/visualization/build_public_reference_viewer.py",
)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def prepare(tmp_path):
    cases, pages, annotations = [], [], []
    for key, case_id in viewer.ROUTES.items():
        cases.append(
            dict(
                id=case_id,
                document_id=key,
                title=key,
                original_pages=[2],
                page_images=[f"cases/{key}/page.png"],
                page_count=1,
            )
        )
        for page in [1, 2]:
            pages.append(dict(case=key, page=page, original_page=page, document_id=key))
            annotations.append(
                dict(
                    id=f"{key}-{page}",
                    case=key,
                    page=page,
                    kind="text",
                    value={"text": "allowed" if page == 2 else "PRIVATE"},
                    private_path="must not publish",
                )
            )
    (tmp_path / "catalog.json").write_text(json.dumps(dict(cases=cases)))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(dict(version="test", pages=pages, annotations=annotations)))
    return frozen


def test_publish_only_allowlisted_annotations(tmp_path):
    frozen = prepare(tmp_path)
    summary = viewer.build(tmp_path, frozen)
    assert summary["pages"] == summary["annotations"] == 4
    text = (tmp_path / "references/annotations.json").read_text()
    assert "PRIVATE" not in text
    assert "private_path" not in text
    data = json.loads(text)
    assert all(a["status"] == "draft" for a in data["annotations"])
    assert [p["case"] for p in data["pages"]] == ["math", "imf", "irena", "nasa"]


def test_reject_mismatched_source_identity(tmp_path):
    frozen = prepare(tmp_path)
    data = json.loads(frozen.read_text())
    data["pages"][1]["document_id"] = "different-source"
    frozen.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source identities"):
        viewer.build(tmp_path, frozen)
