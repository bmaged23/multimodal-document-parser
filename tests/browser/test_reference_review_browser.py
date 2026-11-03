"""Browser review decisions persist to disk and reference text cannot execute script."""

import importlib.util
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from multimodal_doc_parser.evaluation.reference_review import ReviewStore, digest

playwright = pytest.importorskip("playwright.sync_api")


def test_reference_review_draw_edit_approve_export_and_mobile(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (400, 600), "white").save(image)
    dataset = dict(
        pages=[
            dict(
                case="math",
                page=1,
                original_page=1,
                title="Synthetic assignment",
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
                title="<script>window.injected=true</script>",
                value=dict(text="2+2=5", scope="selected_span"),
                source_box=None,
                provenance="Source draft",
                origin=dict(file="synthetic"),
                instruction="Preserve written content",
            )
        ],
    )
    dataset["pages"].append({**dataset["pages"][0], "page": 2, "original_page": 2})
    dataset["annotations"].append(
        {**dataset["annotations"][0], "id": "a2", "page": 2, "title": "Second-page reference"}
    )
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    script = Path("scripts/visualization/serve_reference_review.py")
    spec = importlib.util.spec_from_file_location("reference_review_server", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(module.ReviewHandler, store=ReviewStore(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch(args=["--disable-gpu"])
            page = browser.new_page(viewport=dict(width=1550, height=1050))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            playwright.expect(page.locator("#page-annotations")).to_contain_text("2+2=5")
            assert not page.locator("#edit-panel").get_attribute("open")
            page.locator("#page-next").click()
            playwright.expect(page.locator("#source-title")).to_contain_text("page 2")
            playwright.expect(page.locator("#annotations-title")).to_contain_text("Page 2")
            playwright.expect(page.locator("#page-annotations")).to_contain_text(
                "Second-page reference"
            )
            assert page.locator("#source-image").get_attribute("src").endswith("/2")
            page.locator("#page-previous").click()
            page.get_by_role("button", name="Review / edit", exact=True).click()
            playwright.expect(page.locator("#title")).to_contain_text("<script>")
            assert not page.evaluate("window.injected || false")
            playwright.expect(page.locator("#source-image")).not_to_have_js_property(
                "naturalWidth", 0
            )
            page.locator("#reviewer").fill("Human test reviewer")
            page.locator("#approve").click()
            playwright.expect(page.locator("#save-state")).to_contain_text("Draw a source region")
            svg = page.locator("#selection").bounding_box()
            page.mouse.move(svg["x"] + 40, svg["y"] + 60)
            page.mouse.down()
            page.mouse.move(svg["x"] + 200, svg["y"] + 150)
            page.mouse.up()
            page.locator("#transcription").fill("Exact written text: 2+2=5")
            page.locator("#approve").click()
            playwright.expect(page.locator("#stats")).to_contain_text("1 approved")
            page.reload()
            playwright.expect(page.locator("#status")).to_have_text("approved")
            page.get_by_role("button", name="Review / edit", exact=True).click()
            assert page.locator("#transcription").input_value() == "Exact written text: 2+2=5"
            result = page.request.get(f"http://127.0.0.1:{server.server_port}/api/export").json()
            assert len(result["annotations"]) == 1
            assert result["annotations"][0]["reviewer"] == "Human test reviewer"
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
