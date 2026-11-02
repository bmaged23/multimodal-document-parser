"""Preserve second-model runs under their own namespace and raw checkpoint path."""

import importlib.util
import json
from pathlib import Path


def test_dots_snapshot_retains_correct_manifest_and_runtime(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "archive_runs", "scripts/parser_evaluation/archive_evaluation_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    run = tmp_path / "run"
    raw = run / "raw/dots_mocr"
    raw.mkdir(parents=True)
    checkpoint = {"status": "completed", "pages": [{"page_number": 1}], "backend": "dots.mocr"}
    (raw / "inference-manifest.json").write_text(json.dumps(checkpoint))
    (raw / "page_0001.json").write_text('{"content":"[]"}')
    (run / "run-status.json").write_text('{"status":"completed"}')
    for name in ["document.json", "document.md", "review.html"]:
        (run / name).write_text("test export")
    for name in [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "THIRD_PARTY_NOTICES.md",
        "backends/dots_mocr_runtime/runtime-lock.json",
        "scripts/model_runtimes/serve_dots_mocr.py",
        "scripts/model_runtimes/setup_dots_mocr_runtime.py",
        "scripts/model_runtimes/build_dots_mocr_runtime.py",
        "backends/dots_mocr_runtime/native-build-lock.json",
        "backends/dots_mocr_runtime/toolchain-linux-64.lock",
    ]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("test runtime metadata")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").write_text("")
    case = dict(
        id="example",
        run_directory="run",
        model_name="dots.mocr-Q4_K_M",
        raw_results_subdirectory="dots_mocr",
        evaluation_pages=[1],
        expected_evaluation_pages=1,
    )
    result = module.snapshot(case, tmp_path / "archive")
    archived = tmp_path / result["archive_directory"]
    assert result["checkpointed_pages"] == 1
    assert result["expected_subset_available"]
    assert (
        json.loads((archived / "run/raw/dots_mocr/inference-manifest.json").read_text())
        == checkpoint
    )
    assert not (archived / "run/raw/paddleocr").exists()
    assert (archived / "reproduction/backends/dots_mocr_runtime/runtime-lock.json").exists()
    inventory = json.loads((archived / "file-checksums.json").read_text())
    assert any(file["path"] == "run/raw/dots_mocr/page_0001.json" for file in inventory["files"])
