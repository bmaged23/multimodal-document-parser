"""Create immutable, checksum-inventoried evaluation snapshots and watch final exports."""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "evaluation/preserved-runs.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def raw_subdirectory(case: dict) -> str:
    name = case.get("raw_results_subdirectory", "paddleocr")
    if name not in {"paddleocr", "dots_mocr", "unlimited_ocr_2"}:
        raise ValueError("Unknown raw results subdirectory")
    return name


def progress(case: dict) -> tuple[dict, dict, list[int]]:
    run = ROOT / case["run_directory"]
    inference = read_json(run / "raw" / raw_subdirectory(case) / "inference-manifest.json")
    status = read_json(run / "run-status.json")
    pages = [page["page_number"] for page in inference.get("pages", [])]
    return inference, status, sorted(set(pages))


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot(case: dict, archive_root: Path) -> dict:
    run = ROOT / case["run_directory"]
    inference, status, pages = progress(case)
    state = status.get("status", inference.get("status", "unknown"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model_name = case["model_name"]
    model_directory = f"{model_name}_output"
    if Path(model_directory).name != model_directory:
        raise ValueError("Model name must be a single directory component")
    parent = archive_root / model_directory / case["id"]
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{timestamp}_{state}_{len(pages):02d}_pages"
    staging = Path(tempfile.mkdtemp(prefix=".archiving-", dir=parent))

    def ignore(path: str, names: list[str]) -> set[str]:
        skipped = {name for name in names if name.endswith((".tmp", ".tmp.json"))}
        if Path(path).resolve() == (run / "raw" / raw_subdirectory(case)).resolve():
            for name in names:
                if name.startswith("page_") and name.endswith(".json"):
                    number = name.removeprefix("page_").removesuffix(".json")
                    if number.isdigit() and int(number) not in pages:
                        skipped.add(name)
        return skipped

    try:
        shutil.copytree(run, staging / "run", ignore=ignore)
        # Use the checkpoint captured at the beginning, even if inference advances during copying.
        write_json(
            staging / "run/raw" / raw_subdirectory(case) / "inference-manifest.json", inference
        )
        write_json(staging / "run/run-status.json", status)
        for related in case.get("related_run_directories", []):
            source = ROOT / related
            shutil.copytree(
                source,
                staging / "related-runs" / source.name,
                ignore=shutil.ignore_patterns("*.tmp", "*.tmp.json"),
            )
        reproduction = staging / "reproduction"
        reproduction.mkdir()
        runtime_files = (
            [
                "backends/dots_mocr_runtime/runtime-lock.json",
                "scripts/model_runtimes/serve_dots_mocr.py",
                "scripts/model_runtimes/setup_dots_mocr_runtime.py",
                "scripts/model_runtimes/build_dots_mocr_runtime.py",
                "backends/dots_mocr_runtime/native-build-lock.json",
                "backends/dots_mocr_runtime/toolchain-linux-64.lock",
            ]
            if raw_subdirectory(case) == "dots_mocr"
            else [
                "backends/paddleocr_runtime/pyproject.toml",
                "backends/paddleocr_runtime/uv.lock",
                "backends/paddleocr_runtime/run_inference.py",
            ]
        )
        runtime_files = case.get("reproduction_files", runtime_files)
        for relative in [
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "THIRD_PARTY_NOTICES.md",
            *runtime_files,
        ]:
            source = ROOT / relative
            target = reproduction / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copytree(
            ROOT / "src",
            reproduction / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if case.get("selection_manifest"):
            shutil.copy2(ROOT / case["selection_manifest"], staging / "page-selection.json")
        for key, filename in [
            ("reference_annotations", "reference-annotations.json"),
            ("comparison_report", "comparison.json"),
        ]:
            if case.get(key):
                shutil.copy2(ROOT / case[key], staging / filename)
        input_pdf = staging / "run/source.pdf"
        if input_pdf.exists():
            with pdfium.PdfDocument(input_pdf) as full_source:
                if len(full_source) > len(case["evaluation_pages"]):
                    with pdfium.PdfDocument.new() as selected:
                        selected.import_pages(
                            full_source, [page - 1 for page in case["evaluation_pages"]]
                        )
                        selected.save(staging / "evaluation-input-pages.pdf")
                    write_json(
                        staging / "evaluation-input-page-map.json",
                        {
                            "mapping": [
                                {"subset_page": index + 1, "original_pdf_page": number}
                                for index, number in enumerate(case["evaluation_pages"])
                            ],
                            "note": (
                                "The canonical run retains its original full-source document ID. "
                                "This extracted PDF is a separate evaluation input."
                            ),
                        },
                    )
        metadata = {
            "case": case,
            "model_name": model_name,
            "snapshot_at_utc": timestamp,
            "source_run_status": state,
            "checkpointed_pages": pages,
            "checkpointed_page_count": len(pages),
            "expected_evaluation_pages": case["expected_evaluation_pages"],
            "expected_subset_available": set(case["evaluation_pages"]).issubset(pages),
            "final_exports_available": all(
                (run / name).exists() for name in ["document.json", "document.md", "review.html"]
            ),
            "reference_annotations": case.get("reference_annotations"),
            "code_snapshot_scope": (
                "Current code at archive time; the execution source revision of historical "
                "runs was not pinned."
            ),
            "notes": (
                "Frozen local model predictions, not ground truth. An active-run snapshot may "
                "contain partial logs and prepared-only exports. "
                "Runtime PID files are historical metadata."
            ),
        }
        write_json(staging / "archive-metadata.json", metadata)
        inventory = []
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                with file.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                inventory.append(
                    {
                        "path": file.relative_to(staging).as_posix(),
                        "bytes": file.stat().st_size,
                        "sha256": digest,
                    }
                )
        write_json(staging / "file-checksums.json", {"algorithm": "sha256", "files": inventory})
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    record = {
        "case_id": case["id"],
        "model_name": model_name,
        "archive_directory": str(destination.relative_to(ROOT)),
        "source_status": state,
        "checkpointed_pages": len(pages),
        "expected_pages": case["expected_evaluation_pages"],
        "expected_subset_available": metadata["expected_subset_available"],
        "final_exports_available": metadata["final_exports_available"],
    }
    index_path = archive_root / "latest-snapshots.json"
    index = read_json(index_path)
    if index and index.get("index_version") != "2.0":
        raise ValueError("Archive index needs model-namespace migration")
    index["index_version"] = "2.0"
    index.setdefault("models", {}).setdefault(model_name, {})[case["id"]] = record
    write_json(index_path, index)
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch-case", help="Wait for a case's final exports, then freeze another snapshot"
    )
    parser.add_argument("--case", help="Snapshot one registered case immediately")
    args = parser.parse_args()
    registry = read_json(REGISTRY)
    archive_root = ROOT / registry["archive_root"]
    cases = {case["id"]: case for case in registry["cases"]}
    if args.case and args.case not in cases:
        parser.error("Unknown preservation case")
    if not args.watch_case:
        for case in [cases[args.case]] if args.case else cases.values():
            snapshot(case, archive_root)
        return
    if args.watch_case not in cases:
        parser.error("Unknown preservation case")
    case = cases[args.watch_case]
    print(f"Watching final exports for {case['id']}", flush=True)
    while True:
        inference, status, pages = progress(case)
        state = status.get("status")
        if state in {"failed", "interrupted"}:
            snapshot(case, archive_root)
            return
        if (
            state == "completed"
            and inference.get("status", "completed") == "completed"
            and set(case["evaluation_pages"]).issubset(pages)
        ):
            run = ROOT / case["run_directory"]
            if all(
                (run / name).exists() for name in ["document.json", "document.md", "review.html"]
            ):
                snapshot(case, archive_root)
                return
        time.sleep(30)


if __name__ == "__main__":
    main()
