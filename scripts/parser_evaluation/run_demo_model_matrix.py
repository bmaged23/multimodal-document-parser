"""Reuse the demo's saved OCR and run missing document/model cells separately."""

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("irena", "irena-mixed-content-20-paddleocr-vl-1.6-01", list(range(13, 33))),
    ("imf", "imf-complex-tables-26-paddleocr-vl-1.6-01", list(range(1, 27))),
]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/parser_evaluation" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def missing_cases() -> list[dict]:
    result = []
    for name, paddle, originals in CASES:
        baseline = ROOT / "outputs/parser_runs" / paddle
        document = json.loads((baseline / "document.json").read_text())
        bundle = (
            ROOT
            / "outputs/parsed_documents"
            / (document["document_id"].removeprefix("sha256:") + "-dpi-144")
        )
        if not bundle.exists():
            bundle = prepare_document(baseline / "source.pdf", bundle.parent, dpi=144)
        prepared = json.loads((bundle / "document.json").read_text())
        assert prepared["document_id"] == document["document_id"]
        assert len(prepared["pages"]) == len(originals)
        selection = ROOT / f"evaluation/test_cases/{name}-four-model-demo.json"
        selection.write_text(
            json.dumps(
                {
                    "document_id": prepared["document_id"],
                    "original_pdf_pages": originals,
                    "render_dpi": 144,
                    "notes": "Same retained source selection across four OCR execution profiles.",
                },
                indent=2,
            )
            + "\n"
        )
        result.append(
            {
                "id": f"dots_mocr_q4_{name}_four_model_demo_01",
                "name": name.upper() + " four-model demo",
                "prepared_bundle": str(bundle),
                "run_directory": str(ROOT / f"outputs/parser_runs/dots-mocr-q4-k-m-{name}-demo-01"),
                "total_pages": len(originals),
                "original_pdf_pages": originals,
                "selection_manifest": str(selection.relative_to(ROOT)),
                "paddle_baseline": str(baseline),
                "comparison_directory": str(
                    ROOT / f"outputs/parser_comparisons/{name}-four-model-demo-dots-vs-paddle-01"
                ),
            }
        )
    return result


def resume_dots() -> None:
    module = load_script("run_quantized_evaluation_batch")
    progress = ROOT / "outputs/parser_runs/dots-mocr-q4-evaluation-batch-four-model-demo-01.json"
    batch = module.Batch.__new__(module.Batch)
    batch.suffix = "four-model-demo-01"
    batch.progress = progress
    batch.state = json.loads(progress.read_text())
    batch.state["initial_supervisor_pid"] = batch.state.get("supervisor_pid")
    batch.state["supervisor_pid"] = os.getpid()
    batch.cases = batch.state["cases"]
    batch.model = json.loads((ROOT / "backends/dots_mocr_runtime/runtime-lock.json").read_text())
    batch.server = None
    batch.current = None
    try:
        batch.state["status"] = "running"
        for case in batch.cases:
            if case["status"] == "completed":
                continue
            batch.current = case
            run, bundle = Path(case["run_directory"]), Path(case["prepared_bundle"])
            for source in bundle.rglob("*"):
                if source.is_file():
                    target = run / source.relative_to(bundle)
                    if not target.exists() or target.stat().st_size == 0:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
            settled = []
            for row in case.get("pages", []):
                number = row["page_number"]
                raw = run / "raw/dots_mocr" / f"page_{number:04d}.json"
                try:
                    valid = (
                        row["status"] == "accepted"
                        and json.loads(raw.read_text())["finish_reason"] == "stop"
                    )
                except (OSError, ValueError, KeyError):
                    valid = False
                if valid or row["status"] == "failed":
                    settled.append(row)
                else:
                    case.setdefault("interrupted_attempts", []).append(row)
            case["pages"] = settled
            case["completed_pages"] = sum(r["status"] == "accepted" for r in settled)
            case["failed_page_count"] = sum(r["status"] == "failed" for r in settled)
            case["attempted_pages"] = len(settled)
            case["status"] = "running"
            parent = module.Document.model_validate_json((bundle / "document.json").read_text())
            done = {r["page_number"] for r in settled}
            for number in range(1, case["total_pages"] + 1):
                if number in done:
                    continue
                job = run / "page-jobs" / f"page_{number:04d}"
                if job.exists():
                    raw = run / "raw/dots_mocr" / f"page_{number:04d}.json"
                    if raw.exists():
                        shutil.move(raw, job / "interrupted-parent-page.json")
                    job.rename(job.with_name(job.name + f"-interrupted-{time.time_ns()}"))
                batch.page(case, number, parent)
                os.sync()
            batch.finish_case(case)
        batch.state["status"] = (
            "completed_with_errors"
            if any(c["failed_page_count"] for c in batch.cases)
            else "completed"
        )
    finally:
        batch.stop_server()
        batch.checkpoint()
        os.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "dots", "resume-dots", "unlimited", "prepare-cloud"]
    )
    parser.add_argument("--wait-for-dots", action="store_true")
    args = parser.parse_args()
    if args.wait_for_dots:
        progress = (
            ROOT / "outputs/parser_runs/dots-mocr-q4-evaluation-batch-four-model-demo-01.json"
        )
        while True:
            state = json.loads(progress.read_text())
            if state["status"] in {"completed", "completed_with_errors"}:
                break
            if state["status"] in {"failed", "interrupted"}:
                raise RuntimeError("Dots run must finish before another GPU profile starts")
            time.sleep(15)
    cases = missing_cases()
    if args.stage == "dots":
        load_script("run_quantized_evaluation_batch").Batch("four-model-demo-01").run(cases)
    elif args.stage == "resume-dots":
        resume_dots()
    elif args.stage == "prepare-cloud":
        load_script("prepare_dots_mocr_cloud_test").prepare(
            ROOT / "outputs/maintenance/four-model-demo-colab-inputs-01",
            selections=[
                ("irena-mixed-content-20-pages", "dots_mocr_q4_irena_four_model_demo_01"),
                ("imf-complex-tables-26-pages", "dots_mocr_q4_imf_four_model_demo_01"),
            ],
        )
    elif args.stage == "unlimited":
        selected = [
            dict(
                id=f"unlimited_ocr_2_nf4_{name}_four_model_demo_01",
                name=name.upper() + " four-model demo",
                bundle=Path(case["prepared_bundle"]).name,
                run=f"unlimited-ocr-2-nf4-{name}-demo-01",
                original_pdf_pages=case["original_pdf_pages"],
            )
            for (name, _, _), case in zip(CASES, cases, strict=True)
        ]
        load_script("run_unlimited_ocr_2_evaluation_batch").main(
            cases=selected,
            state_path=ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-four-model-demo-01.json",
        )
    else:
        print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
