"""Verify real saved HTML reports in Chromium; uses the optional browser uv group."""

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

from multimodal_doc_parser.document_schema.document import Document, TableElement


def verify(run: Path, destination: Path) -> dict:
    document = Document.model_validate_json((run / "document.json").read_text())
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(run))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    result = dict(run_directory=str(run), pages=len(document.pages), status="running")
    try:
        with sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html")
            for filename in ["source.pdf", "document.json", "document.md"]:
                assert page.request.get(f"http://127.0.0.1:{server.server_port}/{filename}").ok
            page.get_by_role("checkbox", name="Current page only").check()
            for number in range(1, len(document.pages) + 1):
                page.get_by_role("spinbutton", name="Go to page").fill(str(number))
                page.get_by_role("button", name="Go", exact=True).click()
                assert page.locator("#output-page-label").inner_text() == (
                    f"Page {number} of {len(document.pages)}"
                )
                assert page.locator("#pdf-frame").get_attribute("src") == (
                    f"source.pdf#page={number}"
                )
                for tab, selector in [
                    ("Markdown preview", ".markdown-page"),
                    ("Markdown source", ".source-page"),
                    ("JSON", ".json-page"),
                ]:
                    page.get_by_role("tab", name=tab, exact=True).click()
                    assert page.locator(f"{selector}:visible").count() == 1
                    assert page.locator(f"{selector}:visible").get_attribute("data-page") == str(
                        number
                    )
                page.get_by_role("tab", name="Content", exact=True).click()
                expected = sum(
                    any(r.page_number == number for r in e.source_regions)
                    for e in document.elements
                )
                assert page.locator(".element-card:visible").count() == expected
            page.get_by_role("checkbox", name="Current page only").uncheck()
            # These assertions measure export fidelity to JSON, not source table accuracy.
            merged = 0
            for element in document.elements:
                if not isinstance(element, TableElement):
                    continue
                cells = page.locator(f"#{element.id} table td, #{element.id} table th")
                expected = sorted(element.cells, key=lambda c: (c.row, c.column))
                assert cells.count() == len(expected)
                actual_cells = cells.evaluate_all(
                    "cells => cells.map(e => [e.getAttribute('rowspan'), "
                    "e.getAttribute('colspan'), e.tagName.toLowerCase()])"
                )
                assert actual_cells == [
                    [str(c.row_span), str(c.column_span), "th" if c.is_header else "td"]
                    for c in expected
                ]
                merged += sum(c.row_span > 1 or c.column_span > 1 for c in expected)
            page.get_by_role("tab", name="Markdown preview", exact=True).click()
            page.evaluate(
                "document.querySelectorAll('#preview-pages img').forEach(i => i.loading='eager')"
            )
            page.wait_for_function(
                "[...document.querySelectorAll('#preview-pages img')]"
                ".every(i => i.complete && i.naturalWidth > 0)"
            )
            images = page.locator("#preview-pages img").count()
            page.get_by_role("checkbox", name="Current page only").check()
            chosen = 3 if len(document.pages) == 20 else 13
            page.get_by_role("spinbutton", name="Go to page").fill(str(chosen))
            page.get_by_role("button", name="Go", exact=True).click()
            page.get_by_role("button", name="Page + region", exact=True).click()
            page.wait_for_function("document.getElementById('page-image').naturalWidth > 0")
            page.screenshot(path=str(destination / f"{run.name}.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            result.update(
                status="passed",
                per_page_navigation_and_tabs=len(document.pages),
                table_cells_checked=sum(
                    len(e.cells) for e in document.elements if isinstance(e, TableElement)
                ),
                merged_cells_preserved=merged,
                markdown_images_loaded=images,
                javascript_errors=errors,
                narrow_viewport_passed=True,
            )
            browser.close()
    except Exception as exc:
        result.update(status="failed", error=str(exc), javascript_errors=errors)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    results = [verify(run, args.output_directory) for run in args.run_directories]
    (args.output_directory / "browser-verification.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
