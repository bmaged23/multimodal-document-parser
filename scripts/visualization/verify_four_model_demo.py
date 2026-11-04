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
                        for element in selected["json"].get("elements", []):
                            description = element.get("description")
                            if not description:
                                continue
                            provenance = description["provenance"]
                            model = provenance.get("model_version") or provenance["producer"]
                            producer = (
                                "Gemma" if "gemma" in model.lower() else provenance["producer"]
                            )
                            expect(
                                frame.locator(
                                    f'.gemma-output[data-element="{element["id"]}"] .producer-badge'
                                )
                            ).to_contain_text(producer)
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
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            order = ["handwriting", "imf-pipeline", "irena-pipeline", "nasa-financial"]
            expect(page.locator("#document-select")).to_have_value(order[0])
            assert (
                page.locator("#document-select option").evaluate_all(
                    "options=>options.map(option=>option.value)"
                )
                == order
            )
            positions = {}
            for case_id in order:
                case = next(c for c in catalog["cases"] if c["id"] == case_id)
                page.locator("#document-select").select_option(case_id, force=True)
                expect(page.locator("#page")).to_have_value(str(case["original_pages"][0]))
                expect(page.locator("#run-a")).to_have_value(case["runs"][0])
                assert "-bf16-" in case["runs"][0]
                assert "-dots-" in case["runs"][1]
                assert "-paddle-" in case["runs"][2]
                assert "-unlimited-" in case["runs"][3]
                page.locator("#next").click()
                positions[case_id] = str(case["original_pages"][1])
                expect(page.locator("#page")).to_have_value(positions[case_id])
            for case_id in order:
                page.locator("#document-select").select_option(case_id, force=True)
                expect(page.locator("#page")).to_have_value(positions[case_id])
            for profile in ["dots", "paddle", "unlimited", "bf16"]:
                current_id = page.locator("#document-select").input_value()
                selected_case = next(c for c in catalog["cases"] if c["id"] == current_id)
                selected_run = next(r for r in selected_case["runs"] if f"-{profile}-" in r)
                page.locator("#run-a").select_option(selected_run)
                for case_id in order:
                    page.locator("#document-select").select_option(case_id, force=True)
                    selected_case = next(c for c in catalog["cases"] if c["id"] == case_id)
                    expected_run = next(r for r in selected_case["runs"] if f"-{profile}-" in r)
                    expect(page.locator("#run-a")).to_have_value(expected_run)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            references = json.loads((root / "references/annotations.json").read_text())
            page.set_viewport_size({"width": 1600, "height": 1100})
            page.goto(f"http://127.0.0.1:{server.server_port}/references/")
            expect(page.locator("#document")).to_have_value("math")
            for source in references["pages"]:
                page.locator("#document").select_option(source["case"])
                page.locator("#page").fill(str(source["original_page"]))
                page.locator("#page-form").evaluate("form=>form.requestSubmit()")
                expect(page.locator("#source-title")).to_have_text(
                    f"{source['title']} · original page {source['original_page']}"
                )
                expected = [
                    a
                    for a in references["annotations"]
                    if (a["case"], a["page"]) == (source["case"], source["page"])
                ]
                expect(page.locator("article")).to_have_count(len(expected))
                expect(page.locator("#source")).not_to_have_js_property("naturalWidth", 0)
                if expected and expected[0].get("source_box"):
                    page.locator("article h3").first.click()
                    expect(page.locator("#region")).to_have_attribute("visibility", "visible")
            page.goto(f"http://127.0.0.1:{server.server_port}/references/?document=math&page=1")
            expect(page.locator("#source-title")).to_contain_text("original page 1")
            expect(page.locator("#source")).not_to_have_js_property("naturalWidth", 0)
            source_scroll = page.locator("#source-pane").evaluate("pane=>pane.scrollTop")
            page.locator("#annotation-pane").evaluate("pane=>pane.scrollTop=300")
            assert page.locator("#source-pane").evaluate("pane=>pane.scrollTop") == source_scroll
            page.locator("article h3").last.hover()
            assert page.locator("#source-pane").evaluate("pane=>pane.scrollTop") == source_scroll
            annotation_scroll = page.locator("#annotation-pane").evaluate("pane=>pane.scrollTop")
            page.locator("#source-pane").evaluate("pane=>pane.scrollTop=200")
            assert (
                page.locator("#annotation-pane").evaluate("pane=>pane.scrollTop")
                == annotation_scroll
            )
            target = page.locator(".region-target").first
            annotation_id = target.get_attribute("data-annotation")
            target.dispatch_event("click")
            expect(page.locator("article.active")).to_have_attribute(
                "data-annotation", annotation_id
            )
            expect(page.locator("#region")).to_have_attribute("visibility", "visible")
            page.wait_for_timeout(600)
            assert page.evaluate("""()=>{
                const pane=document.getElementById('annotation-pane');
                const card=document.querySelector('article.active');
                const target=pane.scrollTop+card.getBoundingClientRect().top
                    -pane.getBoundingClientRect().top-pane.querySelector('h2').offsetHeight-12;
                const clamped=Math.max(0,Math.min(pane.scrollHeight-pane.clientHeight,target));
                return Math.abs(pane.scrollTop-clamped)<3;
            }""")
            page.locator("article h3").last.click()
            page.wait_for_timeout(600)
            assert page.evaluate("""()=>{
                const pane=document.getElementById('source-pane');
                const sheet=document.querySelector('.sheet');
                const id=document.querySelector('article.active').dataset.annotation;
                const box=data.annotations.find(a=>a.id===id).source_box;
                const target=pane.scrollTop+sheet.getBoundingClientRect().top
                    -pane.getBoundingClientRect().top+box.y0*sheet.clientHeight
                    -pane.querySelector('h2').offsetHeight-12;
                const clamped=Math.max(0,Math.min(pane.scrollHeight-pane.clientHeight,target));
                return Math.abs(pane.scrollTop-clamped)<3;
            }""")
            page.locator("#annotation-pane").evaluate("pane=>pane.scrollTop=0")
            page.locator("#source-pane").evaluate("pane=>pane.scrollTop=0")
            page.screenshot(path=str(root.parent / "reference-viewer-handwriting.png"))
            table_page = next(p for p in references["pages"] if p["case"] == "imf")
            page.goto(
                f"http://127.0.0.1:{server.server_port}/references/"
                f"?document=imf&page={table_page['original_page']}"
            )
            expect(page.locator("table").first).to_be_attached()
            page.screenshot(path=str(root.parent / "reference-viewer-tables.png"))
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
                document_order=order,
                first_visit_first_page=True,
                per_document_page_memory=True,
                model_selection_preserved_across_documents=True,
                reference_viewer_pages=len(references["pages"]),
                reference_viewer_annotations=len(references["annotations"]),
                reference_viewer_independent_scrolling=True,
                reference_viewer_bidirectional_highlighting=True,
                reference_viewer_click_reveals_opposite_region=True,
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
