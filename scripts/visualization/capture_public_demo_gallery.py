"""Capture five real table/chart/image/text/handwriting previews from the public demo."""

import argparse
import functools
import json
import threading
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from multimodal_doc_parser.evaluation.rag_benchmark import digest

EXAMPLES = [
    ("table", "pipeline", "irena-pipeline", "irena-final", 15, "element_00020"),
    ("chart", "pipeline", "irena-pipeline", "irena-final", 16, "element_00036"),
    ("image", "parsers", "nasa-financial", "nasa-paddle", 4, None),
    ("text", "parsers", "nasa-financial", "nasa-paddle", 6, None),
    ("handwriting", "parsers", "handwriting", "diary-paddle", 6, None),
]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((args.demo_directory / "catalog.json").read_text())
    handler = functools.partial(QuietHandler, directory=str(args.demo_directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = []
    try:
        with sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1800, "height": 1120})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            for category, section, case_id, run_id, number, element_id in EXAMPLES:
                case = next(c for c in catalog["cases"] if c["id"] == case_id)
                number = case["original_pages"].index(number) + 1
                query = urlencode(
                    dict(section=section, case=case_id, run=run_id, page=number, focus=1)
                )
                page.goto(f"http://127.0.0.1:{server.server_port}/index.html?{query}")
                page.wait_for_function(
                    "document.getElementById('loading').textContent.includes('no inference')"
                )
                page.wait_for_function("document.getElementById('source-image').naturalWidth>0")
                case = next(c for c in catalog["cases"] if c["id"] == case_id)
                payload = json.loads(
                    (args.demo_directory / catalog["runs"][run_id]["path"]).read_text()
                )
                data = next(p for p in payload["pages"] if p["number"] == number)
                if element_id is None:
                    kind = "image" if category == "image" else "text"
                    elements = [e for e in data["json"]["elements"] if e["type"] == kind]
                    element = max(elements, key=lambda e: len(e.get("text", "")))
                    element_id = element["id"]
                article = page.frame_locator("#frame-a").locator(
                    f'article[data-element="{element_id}"]'
                )
                article.wait_for()
                page.locator("#view-settings summary").click()
                page.locator("#output-zoom").select_option("125")
                page.locator("#view-settings summary").click()
                article.scroll_into_view_if_needed()
                article.locator("button").click()
                page.wait_for_function("!document.getElementById('source-box').hidden")
                page.wait_for_timeout(300)
                output = args.output_directory / f"demo-{category}.png"
                page.screenshot(path=str(output))
                records.append(
                    dict(
                        file=output.name,
                        category=category,
                        case=case_id,
                        run=run_id,
                        source_element_id=element_id,
                        original_pdf_page=case["original_pages"][number - 1],
                        saved_page_number=data["saved_page_number"],
                        stage=catalog["runs"][run_id]["stage"],
                        source_url=case["source_url"],
                        sha256=digest(output),
                        viewport=dict(width=1800, height=1120),
                    )
                )
            browser.close()
            if errors:
                raise ValueError("Browser errors: " + "; ".join(errors))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    provenance = dict(
        captured_at_utc=datetime.now(UTC).isoformat(),
        note=(
            "Actual browser screenshots of saved public excerpts; "
            "no model inference or image synthesis."
        ),
        demo_build_manifest_sha256=digest(args.demo_directory / "build-manifest.json"),
        presentation="Expanded review; output zoom 125%; selected source region highlighted.",
        images=records,
    )
    (args.output_directory / "readme-gallery.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Captured {len(records)} saved-output previews with source provenance.")


if __name__ == "__main__":
    main()
