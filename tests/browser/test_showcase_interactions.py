"""Exercise showcase model switching, source synchronization and untrusted output."""

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from multimodal_doc_parser.evaluation.showcase import build_showcase


def test_showcase_two_models_failure_pages_formats_and_mobile(showcase_project):
    project, manifest = showcase_project
    fixture = json.loads(manifest.read_text())
    pipeline = json.loads(json.dumps(fixture["cases"][0]))
    pipeline.update(id="fixture-pipeline", title="Synthetic pipeline", group="pipeline")
    pipeline["runs"] = [
        dict(
            id="pipeline-a",
            label="Normalized OCR",
            stage="ocr",
            kind="canonical",
            directory="run-a",
        ),
        dict(
            id="pipeline-b",
            label="Reviewed output",
            stage="source_checked",
            kind="canonical",
            directory="run-b",
        ),
    ]
    fixture["cases"].append(pipeline)
    manifest.write_text(json.dumps(fixture))
    output = project / "showcase"
    build_showcase(project, output, manifest)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(output))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=dict(width=1500, height=1000))
            errors = []
            external = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    external.append(request.url) if "example.com" in request.url else None
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            page.wait_for_function(
                'document.getElementById("loading").textContent.includes("no inference")'
            )
            assert page.locator("#cases .case-button").count() == 2
            assert not page.locator(".metrics").is_visible()
            assert page.locator("#source-pdf").get_attribute("src") is None
            assert "Expand PDF + output" in page.locator("#focus-view").inner_text()
            assert page.locator("#focus-view").get_attribute("aria-controls") == "review-board"
            assert page.locator("#source-viewport").bounding_box()["height"] > 500
            with page.expect_response(
                lambda response: urlsplit(response.url).path.endswith("/source.pdf")
            ):
                page.locator("#source-format").select_option("pdf")
            assert page.locator("#source-pdf").is_visible()
            page.locator("#source-format").select_option("image")
            assert page.locator("#source-pdf").get_attribute("src") is None
            page.locator("#page").fill("15")
            page.locator("#page-form").evaluate("form=>form.requestSubmit()")
            assert page.locator("#page").evaluate("input=>input.validity.customError")
            page.locator("#page").fill("10")
            page.locator("#page-form").evaluate("form=>form.requestSubmit()")
            assert page.locator("#completed").text_content() == "1 / 2"
            page.locator("#compare").check()
            page.wait_for_function('!document.getElementById("pane-b").hidden')
            assert page.locator("#run-b").input_value() == "model-b"
            page.locator("#focus-view").click()
            assert page.locator("body").evaluate("e=>e.classList.contains('focused')")
            page.locator("#run-details summary").click()
            assert page.locator("#evidence-button").is_visible()
            page.locator("#run-details summary").click()
            assert page.locator("#source-viewport").bounding_box()["height"] > 700
            assert page.locator("#pane-a").bounding_box()["width"] > 650
            assert not page.locator("#pane-b").is_visible()
            page.locator("#view-b").click()
            assert page.locator("#pane-b").is_visible()
            page.locator("#view-a").click()
            page.locator("#view-settings summary").click()
            page.locator("#source-zoom").select_option("1.25")
            assert page.locator("#source-zoom").input_value() == "1.25"
            page.locator("#source-zoom").select_option("1")
            page.locator("#output-zoom").select_option("150")
            page.locator("#view-settings summary").click()
            page.locator("#pane-divider").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#pane-divider").get_attribute("aria-valuenow") == "53"
            page.frame_locator("#frame-a").locator("article button").first.click()
            assert page.locator("#source-box").is_visible()
            assert page.evaluate("window.injected") is None
            page.locator("#output-format").select_option("preview")
            page.wait_for_timeout(100)
            page.locator("#output-format").select_option("raw")
            assert "Saved synthetic raw block" in page.locator("#text-a").inner_text()
            page.locator("#page").fill("20")
            page.locator("#page-form").evaluate("(f)=>f.requestSubmit()")
            page.wait_for_function(
                'document.getElementById("page-state").textContent==="Incomplete"'
            )
            assert "Original PDF page 20" in page.locator("#source-mapping").inner_text()
            assert page.locator("#status-a").is_visible()
            page.locator("#run-a").select_option("missing-model")
            page.wait_for_function(
                'document.getElementById("completed").textContent==="Unmeasured"'
            )
            assert not page.locator("#download-a").is_visible()
            page.locator("#document-select").select_option("fixture-pipeline")
            page.wait_for_function(
                "document.getElementById('primary-selector-label').textContent==='Model'"
            )
            assert page.locator("#run-a").input_value() == "pipeline-b"
            page.locator("#document-select").select_option("fixture-public")
            page.wait_for_function(
                "document.getElementById('primary-selector-label').textContent==='Model'"
            )
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            assert not external
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
