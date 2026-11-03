"""Publish read-only references for exactly the source pages in the public demo."""

import hashlib
import json
from pathlib import Path

ROUTES = {
    "math": "handwriting",
    "imf": "imf-pipeline",
    "irena": "irena-pipeline",
    "nasa": "nasa-financial",
}


def build(public: Path, frozen: Path) -> dict:
    dataset = json.loads(frozen.read_text())
    catalog = json.loads((public / "catalog.json").read_text())
    pages, annotations = [], []
    selected = set()
    for key, case_id in ROUTES.items():
        case = next(c for c in catalog["cases"] if c["id"] == case_id)
        for page in dataset["pages"]:
            if page["case"] != key or page["original_page"] not in case["original_pages"]:
                continue
            if page["document_id"] != case["document_id"]:
                raise ValueError("Reference and demo source identities differ")
            index = case["original_pages"].index(page["original_page"])
            selected.add((key, page["page"]))
            pages.append(
                dict(
                    case=key,
                    page=page["page"],
                    original_page=page["original_page"],
                    title=case["title"],
                    image="../" + case["page_images"][index],
                    reading_order=page.get("reading_order", []),
                )
            )
    for annotation in dataset["annotations"]:
        if (annotation["case"], annotation["page"]) in selected:
            annotations.append(
                {
                    k: annotation.get(k)
                    for k in [
                        "id",
                        "case",
                        "page",
                        "title",
                        "kind",
                        "value",
                        "source_box",
                        "provenance",
                        "instruction",
                        "note",
                    ]
                }
                | {"status": annotation.get("status", "draft")}
            )
    if len(selected) != sum(c["page_count"] for c in catalog["cases"]):
        raise ValueError("Every public demo page must have a reference page")
    target = public / "references"
    target.mkdir()
    payload = dict(
        version=dataset["version"],
        frozen_reference_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        scope=(
            "Assistant-created retrospective development references; not independently "
            "verified human ground truth. Public source-page subset only."
        ),
        pages=pages,
        annotations=annotations,
    )
    (target / "annotations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    template = Path(__file__).with_name("public_reference_viewer.html")
    (target / "index.html").write_text(template.read_text())
    return dict(pages=len(pages), annotations=len(annotations), path="references/")
