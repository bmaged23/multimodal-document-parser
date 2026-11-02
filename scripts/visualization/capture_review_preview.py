"""Capture a report preview using the optional browser dependency group."""

import argparse
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(args.run_directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html")
            page.get_by_role("button", name="Page + region").click()
            page.locator("#page-image").wait_for(state="visible")
            page.wait_for_function("document.getElementById('page-image').naturalWidth > 0")
            page.screenshot(path=str(args.run_directory / "review-preview.png"), full_page=True)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
