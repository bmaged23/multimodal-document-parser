"""Run the preserved NASA-20 and diary-5 selections with bounded GPU page jobs."""

import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "backends/unlimited_ocr_2_runtime"
STATE = ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-evaluation-batch-01.json"
CASES = [
    {
        "id": "unlimited_ocr_2_nf4_nasa_first20_01",
        "name": "NASA first 20 pages",
        "bundle": "bc73868c89045132a265a9b6f6aa62b43ce09a8830f7f622c025b2f53a934f66-dpi-144",
        "run": "unlimited-ocr-2-nf4-nasa-first20-01",
        "original_pdf_pages": list(range(1, 21)),
    },
    {
        "id": "unlimited_ocr_2_nf4_diary_five_01",
        "name": "Diary selected five pages",
        "bundle": "221a9811f6739f6788ab8faca2b29bf7b87086a680adc2284188f88d237b4637-dpi-300",
        "run": "unlimited-ocr-2-nf4-diary-five-01",
        "original_pdf_pages": [4, 5, 7, 6, 74],
    },
]


def write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    if STATE.exists():
        raise ValueError("Batch state already exists; use a new batch for another attempt")
    for case in CASES:
        if (ROOT / "outputs/parser_runs" / case["run"]).exists():
            raise ValueError("Output already exists")
    state = {
        "status": "running",
        "supervisor_pid": os.getpid(),
        "total_pages": 25,
        "attempted_pages": 0,
        "completed_pages": 0,
        "failed_pages": 0,
        "settings": {
            "max_new_tokens": 4096,
            "crop_mode": False,
            "temperature": 0,
            "page_job_timeout_seconds_including_load": 720,
            "execution": "sequential independent page processes; model loaded per page",
        },
        "cases": [],
    }
    write(STATE, state)
    for case in CASES:
        bundle = ROOT / "outputs/parsed_documents" / case["bundle"]
        source = json.loads((bundle / "document.json").read_text())
        assert len(source["pages"]) == len(case["original_pdf_pages"])
        run = ROOT / "outputs/parser_runs" / case["run"]
        run.mkdir()
        (run / "page-jobs").mkdir()
        progress = {
            "status": "running",
            "model": "baidu/Unlimited-OCR-2",
            "revision": "c14071b549f4fcab10909f3c4acc6a2edb366947",
            "profile": "NF4 language / FP16 vision / PyTorch attention fallback",
            "id": case["id"],
            "name": case["name"],
            "document_id": source["document_id"],
            "prepared_bundle": str(bundle),
            "run_directory": str(run.relative_to(ROOT)),
            "total_pages": len(source["pages"]),
            "selected_pages": list(range(1, len(source["pages"]) + 1)),
            "original_pdf_pages": case["original_pdf_pages"],
            "completed_pages": 0,
            "attempted_pages": 0,
            "failed_pages": 0,
            "pages": [],
            "settings": state["settings"],
            "notes": "Experimental predictions; generation stop does not validate accuracy.",
        }
        state["cases"].append(progress)
        write(
            run / "source-selection.json",
            {
                "document_id": source["document_id"],
                "original_pdf_pages": case["original_pdf_pages"],
                "prepared_bundle": str(bundle),
            },
        )
        for number in range(1, progress["total_pages"] + 1):
            job = run / "page-jobs" / f"page_{number:04d}"
            progress.update(current_page=number, page_started_at_unix=time.time())
            write(run / "run-status.json", progress)
            write(STATE, state)
            started = time.perf_counter()
            with (run / "page-jobs" / f"page_{number:04d}.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        str(RUNTIME / ".venv/bin/python"),
                        "-u",
                        str(RUNTIME / "run_gpu_probe.py"),
                        "--output-directory",
                        str(job),
                        "--bundle",
                        str(bundle),
                        "--page-numbers",
                        str(number),
                        "--max-new-tokens",
                        "4096",
                    ],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                progress["worker_pid"] = process.pid
                write(run / "run-status.json", progress)
                write(STATE, state)
                try:
                    code = process.wait(timeout=720)
                    error = None
                except subprocess.TimeoutExpired:
                    terminate(process)
                    code, error = -1, "Page job exceeded 720 seconds including model loading"
                finally:
                    terminate(process)
            row = {
                "page_number": number,
                "original_pdf_page": case["original_pdf_pages"][number - 1],
                "exit_code": code,
                "attempt_seconds_including_load": time.perf_counter() - started,
            }
            child = (
                json.loads((job / "run-status.json").read_text())
                if (job / "run-status.json").exists()
                else {}
            )
            prediction = job / f"page_{number:04d}.json"
            if prediction.exists():
                raw = json.loads(prediction.read_text())
                raw["original_pdf_page"] = row["original_pdf_page"]
                write(run / prediction.name, raw)
                shutil.copyfile(
                    job / f"page_{number:04d}-raw.md", run / f"page_{number:04d}-raw.md"
                )
                row.update(
                    finish_reason=raw["finish_reason"],
                    elapsed_seconds=raw["elapsed_seconds"],
                    output_tokens=len(raw["token_ids"]),
                    peak_allocated_mib=raw["peak_allocated_mib"],
                    peak_reserved_mib=raw["peak_reserved_mib"],
                )
            success = code == 0 and row.get("finish_reason") == "stop"
            row["status"] = "completed_generation" if success else "failed"
            if not success:
                row["error"] = (
                    error or child.get("error") or "Incomplete generation; inspect page job"
                )
            row["load_seconds"] = child.get("load_seconds")
            progress["pages"].append(row)
            progress["attempted_pages"] += 1
            progress["completed_pages"] += int(success)
            progress["failed_pages"] += int(not success)
            progress.update(current_page=None, worker_pid=None)
            state["attempted_pages"] += 1
            state["completed_pages"] += int(success)
            state["failed_pages"] += int(not success)
            state["updated_at_utc"] = datetime.now(UTC).isoformat()
            write(run / "run-status.json", progress)
            write(STATE, state)
            print(
                f"{case['name']}: {progress['attempted_pages']}/{progress['total_pages']}; "
                f"{progress['completed_pages']} stopped normally; "
                f"{progress['failed_pages']} failed",
                flush=True,
            )
        progress["status"] = (
            "completed" if not progress["failed_pages"] else "completed_with_errors"
        )
        write(run / "run-status.json", progress)
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/build_unlimited_ocr_2_review.py"),
                str(run),
            ],
            check=True,
        )
        spec = importlib.util.spec_from_file_location(
            "unlimited_archive", ROOT / "scripts/run_unlimited_ocr_2_validation.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.RUN, module.CASE = run, case["id"]
        archive = module.archive()
        progress["archive_directory"] = str(archive.relative_to(ROOT))
        write(STATE, state)
    state["status"] = "completed" if not state["failed_pages"] else "completed_with_errors"
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    write(STATE, state)
    print("GPU batch finished; outputs archived.", flush=True)


if __name__ == "__main__":
    main()
