"""Browser verification for the parent/child inspector using synthetic data."""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("tiktoken")
from PIL import Image
from playwright.sync_api import sync_playwright
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document
from multimodal_doc_parser.rag_preparation.chunk_bundle import prepare_chunks


def test_chunk_preview_source_navigation_search_provenance_and_escaping(tmp_path):
    canvas = Canvas(str(tmp_path / "sample.pdf"))
    canvas.drawString(40, 700, "Synthetic preview")
    canvas.save()
    source = prepare_document(tmp_path / "sample.pdf", tmp_path / "prepared", dpi=72)
    document = Document.model_validate_json((source / "document.json").read_text())
    document.status = "parsed"
    region = [dict(page_number=1, bounding_box=dict(x0=0.1, y0=0.1, x1=0.9, y1=0.9))]
    Image.new("RGB", (20, 20), "navy").save(source / "photo.png")
    document = Document.model_validate(
        {
            **document.model_dump(),
            "elements": [
                dict(
                    id="heading",
                    type="heading",
                    text="Synthetic section",
                    level=1,
                    source_regions=region,
                    provenance=dict(producer="fixture", method="extracted"),
                ),
                dict(
                    id="body",
                    parent_id="heading",
                    type="text",
                    text="Sample narrative. </script><script>window.injected=true</script>",
                    source_regions=region,
                    provenance=dict(producer="fixture", method="extracted"),
                ),
                dict(
                    id="photo",
                    parent_id="heading",
                    type="image",
                    source_regions=region,
                    provenance=dict(producer="fixture", method="extracted"),
                    asset=dict(path="photo.png"),
                    description=dict(
                        text="A synthetic blue square.",
                        provenance=dict(producer="fixture-vlm", method="generated"),
                    ),
                ),
            ],
        }
    )
    (source / "document.json").write_text(document.model_dump_json())
    prepare_chunks(source, tmp_path / "chunks")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/chunks/review.html")
            assert page.locator("#child option").count() == 2
            assert page.evaluate("window.injected") is None
            assert "#page=1" in page.locator("#pdf").get_attribute("src")
            page.locator("#next").click()
            assert "generated" in page.locator("#text").inner_text()
            page.wait_for_function('document.getElementById("crop").naturalWidth>0')
            page.locator("#search").fill("missing-phrase")
            assert "No matching" in page.locator("#text").inner_text()
            page.locator("#search").fill("")
            page.locator("#page").fill("1")
            assert page.locator("#child option").count() == 2
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
