"""Exercise showcase model switching, source synchronization and untrusted output."""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from multimodal_doc_parser.evaluation.showcase import build_showcase


def test_showcase_two_models_failure_pages_formats_and_mobile(showcase_project):
    project, manifest = showcase_project
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
            assert page.locator("#cases .case-button").count() == 1
            assert page.locator("#completed").inner_text() == "1 / 2"
            page.locator("#compare").check()
            page.wait_for_function('!document.getElementById("pane-b").hidden')
            assert page.locator("#run-b").input_value() == "model-b"
            page.locator("#focus-view").click()
            assert page.locator("body").evaluate("e=>e.classList.contains('focused')")
            assert page.locator("#source-viewport").bounding_box()["height"] > 700
            assert page.locator("#pane-a").bounding_box()["width"] > 650
            assert not page.locator("#pane-b").is_visible()
            page.locator("#view-b").click()
            assert page.locator("#pane-b").is_visible()
            page.locator("#view-a").click()
            page.locator("#zoom-in").click()
            assert page.locator("#zoom-value").inner_text() == "125%"
            page.locator("#zoom-fit").click()
            page.locator("#output-zoom").select_option("150")
            page.locator("#pane-divider").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#pane-divider").get_attribute("aria-valuenow") == "53"
            page.frame_locator("#frame-a").locator("article button").first.click()
            assert page.locator("#source-box").is_visible()
            assert page.evaluate("window.injected") is None
            page.locator('[data-mode="preview"]').click()
            page.wait_for_timeout(100)
            page.locator('[data-mode="raw"]').click()
            assert "Saved synthetic raw block" in page.locator("#text-a").inner_text()
            page.locator("#page").fill("2")
            page.locator("#page-form").evaluate("(f)=>f.requestSubmit()")
            page.wait_for_function(
                'document.getElementById("page-state").textContent==="Incomplete"'
            )
            assert "original PDF page 20" in page.locator("#source-mapping").inner_text()
            assert page.locator("#status-a").is_visible()
            page.locator("#run-a").select_option("missing-model")
            page.wait_for_function(
                'document.getElementById("completed").textContent==="Unmeasured"'
            )
            assert not page.locator("#download-a").is_visible()
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            assert not external
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
