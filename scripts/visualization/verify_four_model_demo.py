"""Exercise saved model choices and OCR/Gemma attribution without model inference."""

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-incomplete-local", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    catalog = json.loads((root / "catalog.json").read_text())
    assert len(catalog["cases"]) == 4
    unavailable = [r["id"] for r in catalog["runs"].values() if not r["available"]]
    if not args.allow_incomplete_local:
        assert not unavailable, unavailable
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    checks, errors = [], []
    try:
        with sync_playwright() as manager:
            browser = manager.chromium.launch(args=["--disable-gpu"])
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            for case in catalog["cases"]:
                assert len(case["runs"]) == 4
                for run_id in case["runs"]:
                    meta = catalog["runs"][run_id]
                    if not meta["available"]:
                        continue
                    data = json.loads((root / meta["path"]).read_text())
                    candidates = [p for p in data["pages"] if p["status"] == "accepted"]
                    selected = next(
                        (
                            p
                            for p in candidates
                            if any(e.get("description") for e in p["json"].get("elements", []))
                        ),
                        candidates[0] if candidates else data["pages"][0],
                    )
                    number = selected["number"]
                    url = f"http://127.0.0.1:{server.server_port}/?case={case['id']}&run={run_id}&page={number}"
                    page.goto(url)
                    expect(page.locator("#label-a")).to_contain_text(meta["label"])
                    expect(page.locator("#primary-selector-label")).to_have_text("Model")
                    expect(page.locator("#source-image")).not_to_have_js_property("naturalWidth", 0)
                    frame = page.frame_locator("#frame-a")
                    if selected["status"] == "accepted" and selected["elements"]:
                        expect(frame.locator('[data-producer="ocr"]').first).to_be_visible()
                        expect(
                            frame.locator('[data-producer="ocr"] .producer-badge').first
                        ).to_contain_text(meta["label"])
                    visual_count = sum(
                        bool(e.get("description")) for e in selected["json"].get("elements", [])
                    )
                    if visual_count:
                        expect(frame.locator(".gemma-output").first).to_be_attached()
                        expect(
                            frame.locator(".gemma-output .producer-badge").first
                        ).to_contain_text("Gemma")
                    expect(page.locator("#page")).to_have_value(
                        str(case["original_pages"][number - 1])
                    )
                    checks.append(
                        dict(
                            case=case["id"],
                            run=run_id,
                            original_page=case["original_pages"][number - 1],
                            visible_gemma_cards=visual_count,
                            status="passed",
                        )
                    )
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            dict(
                result="passed",
                checked=checks,
                unavailable=unavailable,
                scope="local incomplete preview" if unavailable else "complete saved demo",
                mobile_width=390,
                javascript_errors=errors,
                inference_requests=0,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(checks)} saved model/document choices; unavailable: {len(unavailable)}")


if __name__ == "__main__":
    main()
