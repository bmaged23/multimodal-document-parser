"""Compare the frozen Q4 GPU and original HF BF16 Colab results without inference."""

import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path

from multimodal_doc_parser.parser_adapters.table_html import table_cells

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "outputs/parser_runs/dots-mocr-original-hf-bf16-colab-31-pages-01"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    manifest = read(BATCH / "remote/input-manifest.json")
    runtime = read(BATCH / "remote/runtime.json")
    remote_progress = read(BATCH / "remote/progress.json")
    registry = read(ROOT / "evaluation/preserved-runs.json")
    cases, viewer = [], []
    urls = {
        "nasa-first-20-pages": "https://www.nasa.gov/budgets-plans-and-reports/agency-financial-reports/",
        "historical-diary-5-pages": "https://npgallery.nps.gov/AssetDetail/539a8c6d-d1c2-40ec-9729-ba874f53ccb1",
    }
    for case in manifest["cases"]:
        baseline = next(c for c in registry["cases"] if c["id"] == case["baseline_case_id"])
        quantized = ROOT / baseline["run_directory"]
        full = ROOT / "outputs/parser_runs" / f"dots-mocr-bf16-colab-{case['id']}-01"
        q = read(quantized / "raw/dots_mocr/inference-manifest.json")
        f = read(full / "raw/dots_mocr/inference-manifest.json")
        status = read(full / "run-status.json")
        generated = next(c for c in remote_progress["cases"] if c["id"] == case["id"])
        durations = [p["elapsed_seconds"] for p in f["pages"]]
        q_durations = [p["elapsed_seconds"] for p in q["pages"]]
        failures = []
        for failure in status["failed_pages"]:
            error = failure["error"]
            reason = "incomplete_generation"
            if "category" in error and "Field required" in error:
                reason = "missing_layout_category"
            elif "positive width" in error:
                reason = "bounding_box_collapses_after_clipping"
            failures.append({"page": failure["page_number"], "reason": reason})
        cases.append(
            {
                "id": case["id"],
                "source_document_id": case["source_document_id"],
                "requested_pages": len(case["pages"]),
                "q4_accepted_pages": q["completed_pages"],
                "bf16_finished_generations": sum(
                    p["status"] == "completed" for p in generated["pages"]
                ),
                "bf16_accepted_pages": f["completed_pages"],
                "bf16_failures": failures,
                "q4_page_median_seconds": statistics.median(q_durations) if q_durations else None,
                "bf16_accepted_page_median_seconds": statistics.median(durations)
                if durations
                else None,
                "baseline_max_tokens": q.get("max_tokens"),
                "bf16_max_tokens": 6144,
                "timing_scope": (
                    "Different hardware/backends; accepted timings exclude loading and failures"
                ),
            }
        )
        viewer.append(
            {
                "id": "precision-" + case["id"],
                "title": case["id"].replace("-", " ").title(),
                "description": (
                    "Matched inputs: Q4/Q8 local GPU versus original HF BF16 Colab T4. "
                    "Raw replies retain strict-format failures."
                ),
                "group": "parsers",
                "private": case["id"].startswith("original"),
                "original_pages": [p["original_pdf_page"] for p in case["pages"]],
                "preview_page": 4 if case["id"].startswith("author") else 1,
                "source_url": urls.get(case["id"]),
                "runs": [
                    {
                        "id": case["id"] + "-q4",
                        "label": "dots.mocr Q4_K_M + Q8_0 · local GTX 1660 Ti",
                        "stage": "ocr",
                        "kind": "canonical",
                        "directory": quantized.name,
                    },
                    {
                        "id": case["id"] + "-bf16",
                        "label": "dots.mocr original BF16 · Colab T4 · SDPA",
                        "stage": "ocr",
                        "kind": "canonical",
                        "directory": full.name,
                    },
                ],
            }
        )
    source_path = ROOT / "scripts/parser_evaluation/evaluate_handwritten_math_comparison.py"
    spec = importlib.util.spec_from_file_location("frozen_handwriting_scorer", source_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.RUNS = {
        "dots.mocr original HF BF16 · Colab T4 · SDPA": (
            "dots-mocr-bf16-colab-author-mathematics-4-pages-01"
        )
    }
    full_math = scorer.evaluate(scorer.REFERENCE)["results"][0]
    baseline_math = read(
        ROOT / "evaluation/results/author-handwritten-mathematics-comparison-v1.json"
    )
    quant_math = next(r for r in baseline_math["results"] if r["model"].startswith("dots"))
    q_checks = {c["id"]: c for c in quant_math["checks"]}
    paired = [
        {
            "id": c["id"],
            "page": c["page"],
            "category": c["category"],
            "q4_passed": q_checks[c["id"]]["passed"],
            "bf16_passed": c["passed"],
        }
        for c in full_math["checks"]
    ]
    sensor_ref = (
        ROOT / "outputs/parser_comparisons/reference_annotations/original-sensor-table.json"
    )
    expected = read(sensor_ref)
    sensor = {}
    private_details = {}
    for label, path in [
        (
            "Q4",
            ROOT
            / "outputs/parser_runs/dots-mocr-q4-k-m-local-validation-02-upstream-prompt"
            / "raw/dots_mocr/page_0001.json",
        ),
        ("BF16", BATCH / "remote/original-multimodal-2-pages/page_0001.json"),
    ]:
        blocks = json.loads(read(path)["content"])
        # Score existing HTML cell text independently of missing layout categories.
        # This does not repair or accept the invalid canonical page.
        tables = [b["text"] for b in blocks if "<table" in str(b.get("text", "")).lower()]
        checks = []
        for item in expected["table_cells"]:
            _, _, cells = table_cells(tables[item["table_index"]])
            actual = next(
                (c for c in cells if c.row == item["row"] and c.column == item["column"]), None
            )
            passed = actual is not None and re.sub(r"\s+", " ", actual.text).strip() == item["text"]
            passed = passed and all(
                getattr(actual, key) == item[key] for key in ["row_span", "column_span"]
            )
            checks.append({"expected": item, "passed": passed})
        sensor[label] = {"passed": sum(c["passed"] for c in checks), "checked": len(checks)}
        private_details[label] = checks
    details = ROOT / "outputs/quality_evaluations/dots-mocr-bf16-colab-comparison-01"
    save(details / "sensor-private-checks.json", private_details)
    save(details / "handwriting-bf16-checks.json", full_math)
    report = {
        "version": "1.0",
        "scope": "Matched-input profile comparison; not a causal quantization experiment",
        "total_pages": 31,
        "bf16_finished_generations": remote_progress["completed_pages"],
        "bf16_accepted_pages": sum(c["bf16_accepted_pages"] for c in cases),
        "matched_input_pngs": sum(len(c["pages"]) for c in manifest["cases"]),
        "cloud_runtime": {k: v for k, v in runtime.items() if k != "limits"},
        "precision_note": (
            "BF16 parameter storage recorded; quantization disabled. Torch support can include "
            "emulation; native BF16 arithmetic is not independently attested."
        ),
        "cases": cases,
        "handwriting": {
            "annotation_status": baseline_math["annotation_status"],
            "reference_sha256": baseline_math["reference_sha256"],
            "evaluator_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "q4": {k: quant_math[k] for k in ["selected_assertions", "categories", "truth_table"]},
            "bf16": {k: full_math[k] for k in ["selected_assertions", "categories", "truth_table"]},
            "paired_checks": paired,
        },
        "sensor_table": {
            "results": sensor,
            "reference_sha256": hashlib.sha256(sensor_ref.read_bytes()).hexdigest(),
            "scope": (
                "20 earlier assistant-reviewed raw cell text/span checks; not full-page accuracy. "
                "BF16 raw cells scored despite whole-page schema rejection."
            ),
        },
        "limitations": [
            "GGUF source revision is not pinned; checkpoint equivalence is unverified.",
            "Hardware, implementation, attention and some token budgets differ.",
            "Selected checks are assistant-reviewed, not human labels or full-document accuracy.",
            "Strict adapter failures and diary timeout retained; no output repaired or rerun.",
        ],
    }
    save(ROOT / "evaluation/results/dots-mocr-colab-bf16-comparison-v1.json", report)
    save(
        ROOT / "evaluation/dots-mocr-precision-showcase-cases.json",
        {"version": "1.0", "cases": viewer},
    )
    print(
        "Handwriting:",
        quant_math["selected_assertions"],
        "versus",
        full_math["selected_assertions"],
    )
    print("Sensor table:", sensor)
    print("Canonical accepted:", report["bf16_accepted_pages"], "/31")


if __name__ == "__main__":
    main()
