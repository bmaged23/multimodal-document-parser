"""Execute an isolated parser, preserve diagnostics, and export review artifacts."""

import json
import shutil
import subprocess
import time
from contextlib import redirect_stdout
from pathlib import Path

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.run_summary import write_run_summary
from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import normalize_dots_mocr_results
from multimodal_doc_parser.parser_adapters.dots_mocr_inference import run_dots_mocr_inference
from multimodal_doc_parser.parser_adapters.paddleocr_adapter import normalize_paddleocr_results
from multimodal_doc_parser.review_interface.html_report import write_html_report


class ParsingRunError(ValueError):
    """An inference or normalization run could not complete."""


def parse_prepared_document(
    prepared_bundle: Path,
    destination: Path,
    runtime_directory: Path,
    *,
    device: str = "gpu:0",
    page_window: int = 1,
    raw_results_directory: Path | None = None,
    backend: str = "paddleocr",
    base_url: str = "http://127.0.0.1:8010/v1",
    model: str = "dots.mocr-Q4_K_M",
    model_version: str | None = None,
    server_kind: str = "llama-cpp",
    timeout_seconds: float = 300,
    max_tokens: int = 8192,
    max_pixels: int = 1_000_000,
    runtime_metadata_path: Path | None = None,
) -> Path:
    if backend not in {"paddleocr", "dots-mocr"}:
        raise ParsingRunError("Unknown parser backend")
    if backend == "dots-mocr" and not model_version:
        raise ParsingRunError("dots.mocr requires an explicit model version/quantization label")
    if backend == "dots-mocr" and page_window != 1:
        raise ParsingRunError("dots.mocr currently supports sequential pages only")
    if page_window not in (1, 2):
        raise ParsingRunError("Page window must be 1 or 2")
    started = time.perf_counter()
    prepared_bundle = prepared_bundle.resolve()
    destination = destination.resolve()
    if destination == prepared_bundle or prepared_bundle in destination.parents:
        raise ParsingRunError("Parser output must be outside the preparation bundle")
    if destination.exists():
        raise ParsingRunError(f"Output already exists: {destination}")
    document = Document.model_validate_json(
        (prepared_bundle / "document.json").read_text(encoding="utf-8")
    )
    if document.status != "prepared":
        raise ParsingRunError("Expected a prepared document bundle")
    interpreter = runtime_directory / ".venv" / "bin" / "python"
    runner = runtime_directory / "run_inference.py"
    if (
        backend == "paddleocr"
        and raw_results_directory is None
        and (not interpreter.is_file() or not runner.is_file())
    ):
        raise ParsingRunError("PaddleOCR runtime is missing; sync backends/paddleocr_runtime first")
    shutil.copytree(prepared_bundle, destination)
    status_path = destination / "run-status.json"
    raw = destination / "raw" / ("paddleocr" if backend == "paddleocr" else "dots_mocr")

    def status(state: str, error: str | None = None) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "status": state,
                    "backend": "PaddleOCR-VL-1.6" if backend == "paddleocr" else "dots.mocr",
                    "mode": "import" if raw_results_directory else "local_inference",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": error,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    status("running")
    try:
        if raw_results_directory is not None:
            shutil.copytree(raw_results_directory, raw)
        elif backend == "dots-mocr":
            metadata = (
                json.loads(runtime_metadata_path.read_text()) if runtime_metadata_path else None
            )
            with (destination / "inference.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log):
                    run_dots_mocr_inference(
                        document,
                        destination,
                        raw,
                        base_url=base_url,
                        model=model,
                        model_version=model_version or "",
                        server_kind=server_kind,
                        timeout_seconds=timeout_seconds,
                        max_tokens=max_tokens,
                        max_pixels=max_pixels,
                        runtime_metadata=metadata,
                    )
        else:
            raw.parent.mkdir(parents=True)
            with (destination / "inference.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        str(interpreter.absolute()),
                        str(runner.resolve()),
                        str(destination),
                        str(raw),
                        "--device",
                        device,
                        "--page-window",
                        str(page_window),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                raise ParsingRunError(
                    f"PaddleOCR exited with code {result.returncode}; "
                    f"see {destination / 'inference.log'}"
                )
        if backend == "paddleocr":
            normalized, warnings = normalize_paddleocr_results(document, destination, raw)
        else:
            normalized, warnings = normalize_dots_mocr_results(
                document, destination, raw, model_version=model_version or ""
            )
        (destination / "document.json").write_text(
            normalized.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (destination / "document.md").write_text(document_markdown(normalized), encoding="utf-8")
        (destination / "normalization-notes.json").write_text(
            json.dumps(warnings, indent=2) + "\n", encoding="utf-8"
        )
        write_html_report(normalized, destination, warnings)
        write_run_summary(normalized, destination, warnings, raw_directory=raw)
        status("completed")
    except Exception as exc:
        status("failed", str(exc))
        raise ParsingRunError(
            f"Parsing failed: {exc}. Diagnostics retained in {destination}"
        ) from exc
    return destination
