"""Cloud imports must preserve failures and never borrow baseline predictions."""

import hashlib
import importlib.util
import json
import zipfile

import pytest


def setup_result(project, *, quantization=None):
    spec = importlib.util.spec_from_file_location(
        "cloud_import", "scripts/parser_evaluation/import_dots_mocr_cloud_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = project
    baseline = project / "outputs/parser_runs/run-a"
    document = json.loads((baseline / "document.json").read_text())
    registry = project / "evaluation/preserved-runs.json"
    registry.parent.mkdir()
    registry.write_text(
        json.dumps({"cases": [{"id": "base", "run_directory": "outputs/parser_runs/run-a"}]})
    )
    inputs = project / "inputs"
    inputs.mkdir()
    (inputs / "run_gpu_test.py").write_text("# synthetic fixture, never executed\n")
    (inputs / "layout-prompt.txt").write_text("Synthetic fixture prompt")
    (inputs / "sent.png").write_bytes(b"synthetic retained input")
    pages = [
        {
            "page_number": n,
            "original_pdf_page": n,
            "input_image": "sent.png",
            "input_image_sha256": hashlib.sha256(b"synthetic retained input").hexdigest(),
            "source_width": p["width_pixels"],
            "source_height": p["height_pixels"],
            "input_width": p["width_pixels"],
            "input_height": p["height_pixels"],
        }
        for n, p in enumerate(document["pages"], 1)
    ]
    manifest = {
        "model_revision": "fixture",
        "cases": [
            {
                "id": "fixture",
                "baseline_case_id": "base",
                "source_document_id": document["document_id"],
                "pages": pages,
            }
        ],
    }
    (inputs / "input-manifest.json").write_text(json.dumps(manifest))
    files = {
        "input-manifest.json": json.dumps(manifest),
        "runtime.json": json.dumps(
            {
                "quantization": quantization,
                "device": "cuda:0",
                "revision": "fixture",
                "weight_dtype": "torch.bfloat16",
                "gpu": "Synthetic GPU",
                "parameter_dtype_counts": {"torch.bfloat16": 1},
            }
        ),
        "fixture/page_0001.json": json.dumps(
            {
                **pages[0],
                "finish_reason": "stop",
                "elapsed_seconds": 1,
                "peak_allocated_mib": 1,
                "content": json.dumps(
                    [{"bbox": [1, 2, 50, 40], "category": "Text", "text": "fresh cloud text"}]
                ),
            }
        ),
        "fixture/page_0002.json": json.dumps(
            {**pages[1], "finish_reason": "length", "content": "["}
        ),
    }
    files["file-checksums.json"] = json.dumps(
        {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
    )
    archive = project / "result.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, value in files.items():
            zipped.writestr(name, value)
    return module, archive, inputs


def test_cloud_import_keeps_partial_page_and_uses_only_cloud_text(showcase_project):
    project, _ = showcase_project
    module, archive, inputs = setup_result(project)
    entries = module.import_results(archive, inputs, project / "cloud-batch")
    run = project / entries[0]["run_directory"]
    document = json.loads((run / "document.json").read_text())
    assert [e["text"] for e in document["elements"]] == ["fresh cloud text"]
    assert not (run / "raw/dots_mocr/page_0002.json").exists()
    assert (run / "raw/dots_mocr/response_0002.json").exists()
    from multimodal_doc_parser.evaluation.showcase import raw_page

    assert raw_page(run, 2, "canonical") == "["
    assert json.loads((run / "run-status.json").read_text())["completed_pages"] == 1
    assert (run / "review.html").exists()


def test_cloud_import_rejects_quantized_runtime(showcase_project):
    project, _ = showcase_project
    module, archive, inputs = setup_result(project, quantization="NF4")
    with pytest.raises(ValueError, match="unquantized"):
        module.import_results(archive, inputs, project / "cloud-batch")
