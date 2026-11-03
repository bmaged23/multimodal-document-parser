"""Optional browser checks: uv run --group browser pytest tests/browser."""

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.parser_adapters.parsing_run import parse_prepared_document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document

playwright = pytest.importorskip("playwright.sync_api")


def test_review_navigation_tabs_search_and_escaping(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 250, "First page")
    canvas.showPage()
    canvas.drawString(10, 250, "Second page")
    canvas.showPage()
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    raw = tmp_path / "raw"
    raw.mkdir()
    original = Path("tests/fixtures/paddleocr/page_0001.json").read_text()
    (raw / "page_0001.json").write_text(original)
    second = json.loads(original)
    second["res"]["parsing_res_list"] = [
        {
            "block_label": "text",
            "block_bbox": [10, 20, 190, 70],
            "block_content": "Second page text",
        }
    ]
    (raw / "page_0002.json").write_text(json.dumps(second))
    parsed = parse_prepared_document(
        prepared, tmp_path / "parsed", Path("unused"), raw_results_directory=raw
    )
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(parsed))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html")
            assert page.locator(".element-card").count() == 9
            assert page.evaluate("window.injected") is None
            assert page.locator("#view-preview").is_visible()
            preview_images = page.locator("#preview-pages img")
            playwright.expect(preview_images).to_have_count(2)
            for image in preview_images.all():
                playwright.expect(image).not_to_have_js_property("naturalWidth", 0)
            assert page.locator('#preview-pages .markdown-page[data-page="1"] img').count() == 2
            page.get_by_role("tab", name="JSON", exact=True).click()
            assert page.locator("#view-json").is_visible()
            page.get_by_role("tab", name="Markdown source", exact=True).click()
            assert page.locator("#view-markdown").is_visible()
            assert "![chart](assets/crops/" in page.locator("#markdown-pages").inner_text()
            page.get_by_role("tab", name="Content", exact=True).click()
            page.get_by_role("searchbox").fill("Latency improved")
            assert page.locator(".element-card:visible").count() == 1
            page.get_by_role("searchbox").fill("")
            page.get_by_role("checkbox", name="Current page only").check()
            page.get_by_role("spinbutton", name="Go to page").fill("2")
            page.get_by_role("button", name="Go", exact=True).click()
            assert page.locator("#output-page-label").inner_text() == "Page 2 of 2"
            assert page.locator(".element-card:visible").count() == 1
            page.get_by_role("button", name="Page 2 ↗").click()
            assert page.locator("#page-viewport").is_visible()
            assert page.locator("#source-box").is_visible()
            assert page.locator("#page-image").get_attribute("src").endswith("page_0002.png")
            page.get_by_role("button", name="PDF", exact=True).click()
            assert page.locator("#pdf-frame").is_visible()
            assert page.locator("#pdf-frame").get_attribute("src") == "source.pdf#page=2"
            page.get_by_role("tab", name="Markdown preview", exact=True).click()
            assert "Second page text" in page.locator("#preview-pages").inner_text()
            assert page.locator('#preview-pages .markdown-page[data-page="1"]').is_hidden()
            page.get_by_role("tab", name="Markdown source", exact=True).click()
            assert "Second page text" in page.locator("#markdown-pages").inner_text()
            page.get_by_role("tab", name="JSON", exact=True).click()
            assert '"page_number": 2' in page.locator("#json-pages").inner_text()
            page.get_by_role("button", name="Previous page").click()
            assert page.locator("#pdf-frame").get_attribute("src") == "source.pdf#page=1"
            assert page.get_by_role("button", name="Previous page").is_disabled()
            page.get_by_role("spinbutton", name="Go to page").fill("99")
            page.get_by_role("button", name="Go", exact=True).click()
            assert page.locator("#pdf-frame").get_attribute("src") == "source.pdf#page=1"
            page.get_by_role("spinbutton", name="Go to page").fill("1")
            page.get_by_role("checkbox", name="Current page only").uncheck()
            assert page.locator("#json-full").is_visible()
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_comparison_loads_both_parser_reports(tmp_path: Path) -> None:
    from multimodal_doc_parser.evaluation.parser_comparison import compare_parser_runs

    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    canvas.drawString(10, 250, "Synthetic report")
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    left = parse_prepared_document(
        prepared,
        tmp_path / "paddle",
        Path("unused"),
        raw_results_directory=Path("tests/fixtures/paddleocr"),
    )
    right = parse_prepared_document(
        prepared,
        tmp_path / "dots",
        Path("unused"),
        backend="dots-mocr",
        model_version="synthetic-fixture",
        raw_results_directory=Path("tests/fixtures/dots_mocr"),
    )
    compare_parser_runs(left, right, tmp_path / "comparison")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://127.0.0.1:{server.server_port}/comparison/comparison.html")
            assert "not scored" in page.locator("body").inner_text()
            left_frame = page.frame_locator('iframe[title="Left parser review"]')
            right_frame = page.frame_locator('iframe[title="Right parser review"]')
            left_frame.get_by_role("tab", name="Content", exact=True).click()
            assert left_frame.locator(".element-card").count() == 8
            right_frame.get_by_role("tab", name="Content", exact=True).click()
            assert right_frame.locator(".element-card").count() == 7
            assert right_frame.locator("body").evaluate("() => window.injected") is None
            page.set_viewport_size({"width": 480, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_finished_partial_report_does_not_claim_pages_are_still_processing(tmp_path: Path) -> None:
    from multimodal_doc_parser.document_schema.document import Document
    from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import normalize_dots_mocr_results
    from multimodal_doc_parser.review_interface.html_report import write_html_report

    source = tmp_path / "two-pages.pdf"
    canvas = Canvas(str(source), pagesize=(200, 300))
    for text in ["first page", "second page"]:
        canvas.drawString(10, 250, text)
        canvas.showPage()
    canvas.save()
    prepared = prepare_document(source, tmp_path / "prepared", dpi=72)
    document = Document.model_validate_json((prepared / "document.json").read_text())
    partial, notes = normalize_dots_mocr_results(
        document,
        prepared,
        Path("tests/fixtures/dots_mocr"),
        model_version="fixture",
        allow_partial=True,
    )
    (prepared / "run-status.json").write_text(
        json.dumps({"status": "completed_with_errors", "completed_pages": 1, "total_pages": 2})
    )
    write_html_report(partial, prepared, notes)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(prepared))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html")
            playwright.expect(page.locator(".progress-notice")).to_contain_text(
                "Finished testing: 1/2 pages accepted"
            )
            assert "Other pages failed" in page.locator(".progress-notice").inner_text()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
