"""Evaluate quantized dots.mocr page by page, preserving successes and failures."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.parser_comparison import compare_parser_runs
from multimodal_doc_parser.evaluation.run_summary import write_run_summary
from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import normalize_dots_mocr_results
from multimodal_doc_parser.parser_adapters.dots_mocr_inference import write_json
from multimodal_doc_parser.parser_adapters.parsing_run import parse_prepared_document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document
from multimodal_doc_parser.review_interface.html_report import write_html_report

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "dots.mocr-Q4_K_M-mmproj-Q8_0"
NASA_BUNDLE = (
    ROOT
    / "outputs/parsed_documents"
    / "65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc-dpi-144"
)
DIARY_BUNDLE = (
    ROOT
    / "outputs/parsed_documents"
    / "221a9811f6739f6788ab8faca2b29bf7b87086a680adc2284188f88d237b4637-dpi-300"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pixels(path: Path) -> str:
    with Image.open(path) as image, image.convert("RGB") as rgb:
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def prepare_existing(source: Path, output: Path, dpi: int) -> Path:
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    candidate = output / f"{digest}-dpi-{dpi}"
    return candidate if candidate.exists() else prepare_document(source, output, dpi=dpi)


def subset_pdf(source: Path, destination: Path, pages: list[int]) -> None:
    with pdfium.PdfDocument(source) as original, pdfium.PdfDocument.new() as subset:
        subset.import_pages(original, [page - 1 for page in pages])
        subset.save(destination)


def prepare_cases(suffix: str) -> list[dict]:
    sample = (
        ROOT / "data/input_documents/evaluation_corpus/nasa_fy2024_financial_first_20_pages.pdf"
    )
    if not sample.exists():
        subset_pdf(NASA_BUNDLE / "source.pdf", sample, list(range(1, 21)))
    nasa = prepare_existing(sample, ROOT / "outputs/parsed_documents", 144)
    parent = Document.model_validate_json((NASA_BUNDLE / "document.json").read_text())
    chosen = Document.model_validate_json((nasa / "document.json").read_text())
    if len(chosen.pages) != 20:
        raise ValueError("NASA evaluation input must have exactly 20 pages")
    for old, new in zip(parent.pages[:20], chosen.pages, strict=True):
        if pixels(NASA_BUNDLE / old.image.path) != pixels(nasa / new.image.path):
            raise ValueError(f"NASA subset pixels differ on page {new.page_number}")
    nasa_selection = ROOT / "evaluation/test_cases/nasa-first-20-dots-mocr.json"
    selection = {
        "original_document_id": parent.document_id,
        "subset_document_id": chosen.document_id,
        "original_pdf_pages": list(range(1, 21)),
        "render_dpi": 144,
        "page_mapping": [{"sample_page": n, "original_pdf_page": n} for n in range(1, 21)],
        "pixel_verification": "Every rendered page matches the original prepared NASA page in RGB.",
        "notes": (
            "All selected pages receive new quantized GPU inference; "
            "no dots.mocr prediction is reused."
        ),
    }
    if not nasa_selection.exists():
        write_json(nasa_selection, selection)
    elif read(nasa_selection) != selection:
        raise ValueError("Existing NASA selection manifest differs")
    return [
        {
            "id": f"dots_mocr_q4_nasa_first20_{suffix}",
            "name": "NASA first 20 pages",
            "prepared_bundle": str(nasa),
            "run_directory": str(
                ROOT / f"outputs/parser_runs/dots-mocr-q4-k-m-nasa-first20-{suffix}"
            ),
            "total_pages": 20,
            "original_pdf_pages": list(range(1, 21)),
            "selection_manifest": str(nasa_selection.relative_to(ROOT)),
            "paddle_baseline": str(
                ROOT / f"outputs/parser_runs/paddleocr-nasa-first20-reference-{suffix}"
            ),
            "comparison_directory": str(
                ROOT / f"outputs/parser_comparisons/nasa-first20-paddleocr-vs-dots-mocr-q4-{suffix}"
            ),
        },
        {
            "id": f"dots_mocr_q4_diary_five_{suffix}",
            "name": "Diary selected five pages",
            "prepared_bundle": str(DIARY_BUNDLE),
            "run_directory": str(
                ROOT / f"outputs/parser_runs/dots-mocr-q4-k-m-diary-five-{suffix}"
            ),
            "total_pages": 5,
            "original_pdf_pages": [4, 5, 7, 6, 74],
            "selection_manifest": (
                "evaluation/test_cases/lucinda-davis-handwriting-replacement-5.json"
            ),
            "paddle_baseline": str(
                ROOT
                / "outputs/parser_runs/lucinda-davis-handwriting-replacement-5-paddleocr-vl-1.6"
            ),
            "comparison_directory": str(
                ROOT / f"outputs/parser_comparisons/diary-five-paddleocr-vs-dots-mocr-q4-{suffix}"
            ),
        },
    ]


def prepare_nasa_reference(case: dict) -> None:
    destination = Path(case["paddle_baseline"])
    if destination.exists():
        raise ValueError(f"Baseline reference output already exists: {destination}")
    staging = Path(case["run_directory"]) / "baseline-import-staging"
    original_run = ROOT / "outputs/parser_runs/nasa-fy2024-financial-paddleocr-vl-1.6"
    shutil.copytree(original_run / "raw/paddleocr", staging)
    previous = read(staging / "inference-manifest.json")
    document = Document.model_validate_json(
        (Path(case["prepared_bundle"]) / "document.json").read_text()
    )
    write_json(
        staging / "inference-manifest.json",
        {
            **previous,
            "document_id": document.document_id,
            "total_pages": 20,
            "completed_pages": 20,
            "status": "completed",
            "total_seconds": None,
            "pages": [p for p in previous["pages"] if p["page_number"] <= 20],
            "reuse_provenance": {
                "original_run_directory": str(original_run),
                "original_document_id": previous["document_id"],
                "original_total_seconds": previous["total_seconds"],
                "new_inference_performed": False,
                "pixel_equivalence_verified": True,
            },
            "memory_measurement": previous.get("memory_measurement", "")
            + " Retained from the original run, not remeasured for this subset reference.",
        },
    )
    parse_prepared_document(
        Path(case["prepared_bundle"]), destination, Path("unused"), raw_results_directory=staging
    )
    shutil.rmtree(staging)


def failed_page_report(run: Path, message: str) -> None:
    response = read(run / "raw/dots_mocr/response_0001.json")
    content = json.dumps(response, indent=2, ensure_ascii=False) if response else message
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Failed page diagnostics</title>
<style>body{{font:14px system-ui;margin:0}}
header{{padding:16px;background:#fff1d6}}
main{{display:grid;grid-template-columns:1fr 1fr;height:85vh}}
iframe{{width:100%;height:100%;border:0}}
pre{{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:18px}}</style>
<header><strong>Failed page — no accepted extraction</strong><p>{escape(message)}</p></header>
<main><iframe src="source.pdf#page=1" title="Original PDF page"></iframe>
<pre>{escape(content)}</pre></main></html>"""
    (run / "diagnostic-review.html").write_text(html)


class Batch:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.progress = ROOT / f"outputs/parser_runs/dots-mocr-q4-evaluation-batch-{suffix}.json"
        if self.progress.exists():
            raise ValueError(f"Batch progress already exists: {self.progress}; use a fresh suffix")
        self.model = read(ROOT / "backends/dots_mocr_runtime/runtime-lock.json")
        self.server = None
        self.current = None
        self.cases = []
        self.state = {
            "status": "preparing",
            "model_name": MODEL_NAME,
            "model_version": self.model["model_version"],
            "device": "gpu:0",
            "supervisor_pid": os.getpid(),
            "total_pages": 25,
            "max_pixels": 1_000_000,
            "max_tokens": 6144,
            "page_timeout_seconds": 300,
            "cases": self.cases,
        }

    def checkpoint(self) -> None:
        self.state["updated_at_utc"] = datetime.now(UTC).isoformat()
        self.state["attempted_pages"] = sum(c.get("attempted_pages", 0) for c in self.cases)
        self.state["accepted_pages"] = sum(c.get("completed_pages", 0) for c in self.cases)
        self.state["failed_pages"] = sum(c.get("failed_page_count", 0) for c in self.cases)
        write_json(self.progress, self.state)
        write_json(
            ROOT / "outputs/parser_runs/active-run.json",
            {
                "backend": "dots.mocr",
                "device": "gpu:0",
                "model_profile": "gpu-q4",
                "status": self.state["status"],
                "progress_file": str(self.progress),
                "server_pid": self.server.pid if self.server else None,
                "current_case": self.current["id"] if self.current else None,
                "run_directory": self.current["run_directory"] if self.current else None,
            },
        )
        if self.current:
            run = Path(self.current["run_directory"])
            write_json(run / "run-status.json", self.current)

    def stop_server(self) -> None:
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.server = None

    def ensure_server(self) -> None:
        if self.server and self.server.poll() is None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8010/health", timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
        self.stop_server()
        log = (ROOT / f"outputs/dots-mocr-q4-evaluation-server-{self.suffix}.log").open("a")
        self.server = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/model_runtimes/serve_dots_mocr.py"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.checkpoint()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("GPU server exited during startup")
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8010/v1/models", timeout=3
                ) as response:
                    available = json.load(response)
                if self.model["server_model_alias"] in [m["id"] for m in available["data"]]:
                    return
            except Exception:
                time.sleep(1)
        raise TimeoutError("GPU server startup exceeded two minutes")

    def page(self, case: dict, number: int, parent: Document) -> None:
        run = Path(case["run_directory"])
        job = run / "page-jobs" / f"page_{number:04d}"
        job.mkdir(parents=True)
        row = {
            "page_number": number,
            "original_pdf_page": case["original_pdf_pages"][number - 1],
            "status": "running",
            "job_directory": str(job.relative_to(run)),
            "inference_attempted": False,
        }
        case["pages"].append(row)
        case["current_page"] = number
        self.checkpoint()
        started = time.perf_counter()
        try:
            subset_pdf(run / "source.pdf", job / "input.pdf", [number])
            bundle = prepare_document(
                job / "input.pdf", job / "prepared", dpi=parent.preparation.dpi
            )
            single = Document.model_validate_json((bundle / "document.json").read_text())
            if pixels(run / parent.pages[number - 1].image.path) != pixels(
                bundle / single.pages[0].image.path
            ):
                raise ValueError("Single-page rendering differs from selected source page")
            self.ensure_server()
            row["inference_attempted"] = True
            result = parse_prepared_document(
                bundle,
                job / "run",
                Path("unused"),
                backend="dots-mocr",
                model=self.model["server_model_alias"],
                model_version=self.model["model_version"],
                timeout_seconds=300,
                max_tokens=6144,
                max_pixels=1_000_000,
                runtime_metadata_path=ROOT / ".cache/dots_mocr/server-launch.json",
            )
            raw = read(result / "raw/dots_mocr/page_0001.json")
            raw["page_number"] = number
            raw["page_job_provenance"] = {
                "job_directory": str(result.relative_to(run)),
                "single_page_document_id": single.document_id,
                "parent_document_id": parent.document_id,
                "parent_page_number": number,
                "mapping_note": (
                    "Unmodified response content/input hash; page envelope mapped to parent page."
                ),
            }
            write_json(run / "raw/dots_mocr" / f"page_{number:04d}.json", raw)
            row["status"] = "accepted"
            row["inference"] = read(result / "raw/dots_mocr/inference-manifest.json")
            case["completed_pages"] += 1
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            result = job / "run"
            row["inference"] = read(result / "raw/dots_mocr/inference-manifest.json")
            if result.exists():
                failed_page_report(result, str(exc))
            case["failed_page_count"] += 1
        finally:
            row["attempt_seconds"] = time.perf_counter() - started
            case["attempted_pages"] += 1
            self.checkpoint()
        print(
            f"{case['name']}: attempted {case['attempted_pages']}/{case['total_pages']}; "
            f"accepted {case['completed_pages']}, failed {case['failed_page_count']}",
            flush=True,
        )

    def finish_case(self, case: dict, interrupted: bool = False) -> None:
        run = Path(case["run_directory"])
        parent = Document.model_validate_json(
            (Path(case["prepared_bundle"]) / "document.json").read_text()
        )
        state = (
            "interrupted"
            if interrupted
            else ("completed_with_errors" if case["failed_page_count"] else "completed")
        )
        case.update(status=state, current_page=None)
        self.checkpoint()
        timings = []
        for item in case["pages"]:
            if item["status"] == "accepted":
                original = item["inference"]["pages"][0]
                timings.append(
                    {
                        **original,
                        "page_number": item["page_number"],
                        "original_pdf_page": item["original_pdf_page"],
                    }
                )
        manifest = {
            "status": state,
            "backend": "dots.mocr",
            "document_id": parent.document_id,
            "model": self.model["server_model_alias"],
            "model_version": self.model["model_version"],
            "runtime_metadata": read(ROOT / ".cache/dots_mocr/server-launch.json"),
            "pages": timings,
            "completed_pages": len(timings),
            "total_pages": len(parent.pages),
            "attempted_pages": case["attempted_pages"],
            "failed_pages": [
                {k: v for k, v in item.items() if k != "inference"}
                for item in case["pages"]
                if item["status"] != "accepted"
            ],
            "max_tokens": 6144,
            "max_pixels": 1_000_000,
            "temperature": 0.0,
            "peak_sampled_device_memory_mib": max(
                (
                    item.get("inference", {}).get("peak_sampled_device_memory_mib") or 0
                    for item in case["pages"]
                ),
                default=0,
            ),
            "total_seconds": sum(
                item.get("inference", {}).get("total_seconds", 0) for item in case["pages"]
            ),
            "measurement_scope": (
                "Sum of isolated per-page HTTP-client measurements. Same shared GPU server; "
                "server startup excluded. Accepted envelopes map single-page jobs "
                "to parent pages. Original job responses/logs retained under page-jobs."
            ),
        }
        write_json(run / "raw/dots_mocr/inference-manifest.json", manifest)
        normalized, notes = normalize_dots_mocr_results(
            parent,
            run,
            run / "raw/dots_mocr",
            model_version=self.model["model_version"],
            allow_partial=True,
        )
        notes += [
            f"Page {row['page_number']} failed: {row['error']}"
            for row in case["pages"]
            if row["status"] == "failed"
        ]
        write_json(run / "document.json", normalized.model_dump())
        (run / "document.md").write_text(document_markdown(normalized))
        write_json(run / "normalization-notes.json", notes)
        write_run_summary(normalized, run, notes, raw_directory=run / "raw/dots_mocr")
        write_html_report(normalized, run, notes)
        accepted = [row["page_number"] for row in case["pages"] if row["status"] == "accepted"]
        if accepted:
            compare_parser_runs(
                Path(case["paddle_baseline"]),
                run,
                Path(case["comparison_directory"]),
                pages=accepted,
            )
        registry_path = ROOT / "evaluation/preserved-runs.json"
        registry = read(registry_path)
        entry = {
            "id": case["id"],
            "model_name": MODEL_NAME,
            "run_directory": str(run.relative_to(ROOT)),
            "raw_results_subdirectory": "dots_mocr",
            "expected_evaluation_pages": case["total_pages"],
            "evaluation_pages": list(range(1, case["total_pages"] + 1)),
            "selection_manifest": case["selection_manifest"],
            "notes": (
                "New quantized GPU inference on every requested page. Isolated page jobs "
                "preserve failures; only accepted pages appear in canonical output. "
                "Missing pages are not completed extractions. Comparisons cover accepted "
                "matching pages only."
            ),
        }
        if accepted:
            entry["comparison_report"] = str(
                (Path(case["comparison_directory"]) / "comparison.json").relative_to(ROOT)
            )
        if not any(item["id"] == entry["id"] for item in registry["cases"]):
            registry["cases"].append(entry)
            write_json(registry_path, registry)
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/parser_evaluation/archive_evaluation_runs.py"),
                "--case",
                case["id"],
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"Saved {case['name']} report: {run / 'review.html'}", flush=True)

    def run(self) -> None:
        self.checkpoint()
        try:
            self.cases.extend(prepare_cases(self.suffix))
            for case in self.cases:
                run = Path(case["run_directory"])
                if run.exists():
                    raise ValueError(f"Output exists: {run}")
                shutil.copytree(Path(case["prepared_bundle"]), run)
                (run / "raw/dots_mocr").mkdir(parents=True)
                case.update(
                    status="queued",
                    attempted_pages=0,
                    completed_pages=0,
                    failed_page_count=0,
                    pages=[],
                )
                write_json(run / "run-status.json", case)
                write_json(
                    run / "source-selection.json",
                    {
                        "original_pdf_pages": case["original_pdf_pages"],
                        "prepared_bundle": case["prepared_bundle"],
                        "notes": "All selected pages receive new quantized GPU inference.",
                    },
                )
            self.current = self.cases[0]
            prepare_nasa_reference(self.current)
            self.state["status"] = "running"
            for case in self.cases:
                self.current = case
                case["status"] = "running"
                self.checkpoint()
                parent = Document.model_validate_json(
                    (Path(case["prepared_bundle"]) / "document.json").read_text()
                )
                for number in range(1, case["total_pages"] + 1):
                    self.page(case, number, parent)
                self.finish_case(case)
            self.state["status"] = (
                "completed_with_errors"
                if any(c["failed_page_count"] for c in self.cases)
                else "completed"
            )
        except KeyboardInterrupt:
            self.state["status"] = "interrupted"
            if self.current and self.current["status"] == "running":
                self.finish_case(self.current, interrupted=True)
        except Exception as exc:
            self.state.update(status="failed", error=str(exc))
            raise
        finally:
            self.stop_server()
            self.checkpoint()
            print("Quantized GPU batch stopped; checkpoints and outputs retained.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="01")
    args = parser.parse_args()
    os.chdir(ROOT)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    Batch(args.suffix).run()


if __name__ == "__main__":
    main()
