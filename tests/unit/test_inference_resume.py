"""Verify checkpoint reuse without loading real GPU models."""

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def test_resume_skips_completed_pages_and_preserves_history(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pages = [
        {
            "page_number": number,
            "width_pixels": 100,
            "height_pixels": 200,
            "image": {"path": f"page_{number:04d}.png"},
        }
        for number in [1, 2]
    ]
    (bundle / "document.json").write_text(json.dumps({"document_id": "test", "pages": pages}))
    raw = tmp_path / "raw"
    raw.mkdir()
    first = json.dumps({"width": 100, "height": 200})
    (raw / "page_0001.json").write_text(first)
    (raw / "inference-manifest.json").write_text(
        json.dumps(
            {
                "document_id": "test",
                "device": "cpu",
                "paddleocr_version": "test",
                "paddlepaddle_version": "test",
                "total_seconds": 1.0,
                "initialization_seconds": 0.5,
                "pages": [{"page_number": 1, "elapsed_seconds": 0.5}],
            }
        )
    )
    calls = []

    class Result:
        def __init__(self, path):
            self.json = {"input_path": path, "width": 100, "height": 200}

        def save_to_json(self, path):
            Path(path).write_text(json.dumps(self.json))

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def predict(self, path):
            calls.append(Path(path).name)
            yield Result(path)

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=Pipeline))
    monkeypatch.setattr(
        sys, "argv", ["run_inference.py", str(bundle), str(raw), "--device", "cpu", "--resume"]
    )
    runpy.run_path("backends/paddleocr_runtime/run_inference.py", run_name="__main__")
    assert calls == ["page_0002.png"]
    assert (raw / "page_0001.json").read_text() == first
    manifest = json.loads((raw / "inference-manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_pages"] == 2
    assert len(list(raw.glob("inference-manifest.before-resume-*.json"))) == 1
