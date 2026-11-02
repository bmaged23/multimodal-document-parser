"""Verify Colab output and export separate canonical runs without new inference."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.run_summary import write_run_summary
from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import (
    decode_layout,
    normalize_dots_mocr_results,
)
from multimodal_doc_parser.review_interface.html_report import write_html_report

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def import_results(archive: Path, inputs: Path, destination: Path) -> list[dict]:
    archive, inputs, destination = archive.resolve(), inputs.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(archive, destination / "original-colab-results.zip")
    remote = destination / "remote"
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe result ZIP member")
        zipped.extractall(remote)
    for relative, expected in read(remote / "file-checksums.json").items():
        path = (remote / relative).resolve()
        if not path.is_relative_to(remote.resolve()):
            raise ValueError("Unsafe checksum path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Remote output checksum mismatch: " + relative)
    manifest, runtime = read(remote / "input-manifest.json"), read(remote / "runtime.json")
    if manifest != read(inputs / "input-manifest.json"):
        raise ValueError("Cloud input selection differs from preserved matched inputs")
    if runtime["quantization"] is not None or runtime["device"] != "cuda:0":
        raise ValueError("Expected unquantized cloud GPU execution")
    if runtime["revision"] != manifest["model_revision"]:
        raise ValueError("Unexpected model revision")
    if not set(runtime["parameter_dtype_counts"]).issubset(
        {"torch.bfloat16", "torch.float16", "torch.float32"}
    ):
        raise ValueError("Unexpected non-floating model parameters")
    precision = "bf16" if runtime["weight_dtype"] == "torch.bfloat16" else "fp16"
    version = (
        f"dots.mocr original HF@{runtime['revision']}; {precision.upper()}; "
        f"Colab {runtime['gpu']}; SDPA"
    )
    registry_path = ROOT / "evaluation/preserved-runs.json"
    registry = read(registry_path)
    added = []
    for case in manifest["cases"]:
        baseline = next(c for c in registry["cases"] if c["id"] == case["baseline_case_id"])
        source = ROOT / baseline["run_directory"]
        parent = Document.model_validate_json((source / "document.json").read_text())
        parent = parent.model_copy(update={"elements": [], "status": "prepared"})
        if parent.document_id != case["source_document_id"]:
            raise ValueError("Source document differs from quantized baseline")
        run = ROOT / "outputs/parser_runs" / f"dots-mocr-{precision}-colab-{case['id']}-01"
        run.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / "source.pdf", run / "source.pdf")
        for page in parent.pages:
            target = run / page.image.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / page.image.path, target)
        raw = run / "raw/dots_mocr"
        raw.mkdir(parents=True)
        accepted, failures = [], []
        for page in case["pages"]:
            number = page["page_number"]
            sent_image = inputs / page["input_image"]
            if hashlib.sha256(sent_image.read_bytes()).hexdigest() != page["input_image_sha256"]:
                raise ValueError("Preserved model-input PNG changed")
            (raw / "input-images").mkdir(exist_ok=True)
            shutil.copyfile(sent_image, raw / "input-images" / f"page_{number:04d}.png")
            prediction = read(remote / case["id"] / f"page_{number:04d}.json")
            save(raw / f"response_{number:04d}.json", prediction)
            try:
                if prediction.get("finish_reason") != "stop":
                    raise ValueError(
                        prediction.get("error") or "Generation did not finish normally"
                    )
                decode_layout(prediction["content"])
                if prediction["input_image_sha256"] != page["input_image_sha256"]:
                    raise ValueError("Input image hash differs")
                envelope = {
                    "format_version": "dots-mocr-page-v1",
                    **prediction,
                    "coordinate_space": "input_pixels",
                }
                validation = raw / "validation" / f"page_{number:04d}"
                save(validation / f"page_{number:04d}.json", envelope)
                normalize_dots_mocr_results(
                    parent, run, validation, model_version=version, allow_partial=True
                )
                save(raw / f"page_{number:04d}.json", envelope)
                accepted.append(
                    {
                        "page_number": number,
                        "elapsed_seconds": prediction["elapsed_seconds"],
                        "peak_allocated_mib": prediction["peak_allocated_mib"],
                    }
                )
            except (ValueError, KeyError) as exc:
                failures.append({"page_number": number, "error": str(exc)})
        save(raw / "runtime.json", runtime)
        save(
            raw / "inference-manifest.json",
            {
                "document_id": parent.document_id,
                "backend": "dots.mocr",
                "model_version": version,
                "status": "completed_with_errors" if failures else "completed",
                "pages": accepted,
                "total_pages": len(case["pages"]),
                "completed_pages": len(accepted),
                "failed_pages": failures,
                "runtime_metadata": runtime,
                "max_tokens": 6144,
                "max_pixels": 1_000_000,
                "temperature": 0,
                "memory_measurement": (
                    "Cloud PyTorch allocator peak; not local device-total sampling"
                ),
                "measurement_scope": (
                    "Cloud page timings exclude loading/upload; hardware and backends differ"
                ),
            },
        )
        document, notes = normalize_dots_mocr_results(
            parent, run, raw, model_version=version, allow_partial=True
        )
        notes += [f"Page {f['page_number']}: {f['error']}" for f in failures]
        save(run / "document.json", document.model_dump(mode="json"))
        (run / "document.md").write_text(document_markdown(document))
        save(run / "normalization-notes.json", notes)
        write_run_summary(document, run, notes, raw_directory=raw)
        write_html_report(document, run, notes)
        save(
            run / "run-status.json",
            {
                "status": "completed_with_errors" if failures else "completed",
                "completed_pages": len(accepted),
                "total_pages": len(case["pages"]),
                "attempted_pages": len(case["pages"]),
                "failed_pages": failures,
            },
        )
        entry = {
            "id": f"dots_mocr_original_hf_{precision}_colab_{case['id']}_01",
            "model_name": f"dots.mocr-original-HF-{precision.upper()}-Colab-T4-SDPA",
            "run_directory": str(run.relative_to(ROOT)),
            "raw_results_subdirectory": "dots_mocr",
            "expected_evaluation_pages": len(case["pages"]),
            "evaluation_pages": list(range(1, len(case["pages"]) + 1)),
            "related_run_directories": [str(destination.relative_to(ROOT))],
            "reproduction_files": [
                str((inputs / "run_gpu_test.py").relative_to(ROOT)),
                str((inputs / "layout-prompt.txt").relative_to(ROOT)),
                "scripts/parser_evaluation/import_dots_mocr_cloud_results.py",
            ],
            "notes": (
                "Original pinned HF floating weights on Colab GPU; "
                "source-lineage and runtime confounds retained."
            ),
        }
        if baseline.get("selection_manifest"):
            entry["selection_manifest"] = baseline["selection_manifest"]
        added.append(entry)
        registry["cases"].append(entry)
        save(registry_path, registry)
        print(f"{case['id']}: {len(accepted)}/{len(case['pages'])} accepted; {run / 'review.html'}")
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    import_results(args.results_zip, args.inputs, args.output_directory)
