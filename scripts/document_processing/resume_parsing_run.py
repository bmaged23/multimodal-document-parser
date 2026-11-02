"""Resume an interrupted single-model run and refresh a clearly marked live report."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.run_summary import write_run_summary
from multimodal_doc_parser.parser_adapters.paddleocr_adapter import normalize_paddleocr_results
from multimodal_doc_parser.review_interface.html_report import write_html_report


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_bundle", type=Path)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    prepared = Document.model_validate_json(
        (args.prepared_bundle / "document.json").read_text(encoding="utf-8")
    )
    existing = Document.model_validate_json(
        (args.run_directory / "document.json").read_text(encoding="utf-8")
    )
    if prepared.document_id != existing.document_id or prepared.pages != existing.pages:
        parser.error("Prepared input and interrupted run do not match")
    runtime = Path(__file__).resolve().parents[2] / "backends/paddleocr_runtime"
    raw = args.run_directory / "raw/paddleocr"
    manifest_path = raw / "inference-manifest.json"
    checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint.get("status") not in {"failed", "interrupted"}:
        parser.error("Only a stopped run can be resumed")

    def export(partial: bool) -> None:
        document, notes = normalize_paddleocr_results(
            prepared, args.run_directory, raw, allow_partial=partial
        )
        write_json(args.run_directory / "document.json", document.model_dump())
        temporary = args.run_directory / "document.md.tmp"
        temporary.write_text(document_markdown(document), encoding="utf-8")
        temporary.replace(args.run_directory / "document.md")
        write_json(args.run_directory / "normalization-notes.json", notes)
        write_html_report(document, args.run_directory, notes)
        write_run_summary(document, args.run_directory, notes)

    def status(state: str, count: int, error: str | None = None) -> None:
        write_json(
            args.run_directory / "run-status.json",
            {
                "status": state,
                "mode": "local_inference",
                "backend": "PaddleOCR-VL-1.6",
                "page_window": 1,
                "completed_pages": count,
                "total_pages": len(prepared.pages),
                "error": error,
            },
        )

    export(partial=True)
    status("running", checkpoint["completed_pages"])
    command = [
        str(runtime / ".venv/bin/python"),
        str(runtime / "run_inference.py"),
        str(args.run_directory.resolve()),
        str(raw.resolve()),
        "--device",
        "gpu:0",
        "--page-window",
        "1",
        "--resume",
    ]
    with (args.run_directory / "inference.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        last_completed = checkpoint["completed_pages"]
        print(
            f"Resumed original pipeline from {last_completed}/{len(prepared.pages)} pages",
            flush=True,
        )
        while process.poll() is None:
            try:
                current = json.loads(manifest_path.read_text(encoding="utf-8"))
                count = current["completed_pages"]
                if count != last_completed:
                    export(partial=True)
                    status("running", count)
                    last_completed = count
                    print(f"Live report updated: {count}/{len(prepared.pages)} pages", flush=True)
            except (ValueError, OSError) as exc:
                print(f"Preview refresh deferred: {exc}", flush=True)
            time.sleep(2)
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = current["completed_pages"]
        if process.returncode:
            status("failed", count, f"Inference exited with code {process.returncode}")
            raise SystemExit(process.returncode)
        export(partial=False)
        status("completed", len(prepared.pages))
        print(args.run_directory / "review.html", flush=True)


if __name__ == "__main__":
    main()
