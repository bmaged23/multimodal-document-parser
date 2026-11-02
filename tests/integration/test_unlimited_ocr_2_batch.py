"""A failed isolated GPU job must not prevent the next selected page being tested."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_failed_job_is_retained_and_following_page_completes(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "unlimited_batch", "scripts/run_unlimited_ocr_2_evaluation_batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.RUNTIME = tmp_path / "runtime"
    module.STATE = tmp_path / "outputs/parser_runs/state.json"
    module.STATE.parent.mkdir(parents=True)
    bundle = tmp_path / "outputs/parsed_documents/fixture"
    bundle.mkdir(parents=True)
    (bundle / "document.json").write_text(json.dumps({"document_id": "fixture", "pages": [{}, {}]}))
    module.CASES = [
        dict(
            id="fixture",
            name="fixture",
            bundle="fixture",
            run="fixture",
            original_pdf_pages=[4, 74],
        )
    ]
    attempts = []

    class FakeJob:
        pid = 123

        def __init__(self, command, **kwargs):
            number = int(command[command.index("--page-numbers") + 1])
            attempts.append(number)
            self.number = number
            job = Path(command[command.index("--output-directory") + 1])
            job.mkdir()
            (job / "run-status.json").write_text(
                json.dumps(
                    {"load_seconds": 1, **({"error": "fixture error"} if number == 1 else {})}
                )
            )
            if number == 2:
                raw = dict(
                    finish_reason="stop",
                    elapsed_seconds=2,
                    token_ids=[1, 2],
                    peak_allocated_mib=3,
                    peak_reserved_mib=4,
                )
                (job / "page_0002.json").write_text(json.dumps(raw))
                (job / "page_0002-raw.md").write_text("fixture transcription")

        def wait(self, **kwargs):
            return 1 if self.number == 1 else 0

        def poll(self):
            return self.wait()

    monkeypatch.setattr(module.subprocess, "Popen", FakeJob)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: None)
    fake_archive = SimpleNamespace(archive=lambda: tmp_path / "archive")
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None))
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *a: fake_spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda *a: fake_archive)
    module.main()
    state = json.loads(module.STATE.read_text())
    assert attempts == [1, 2]
    assert state["status"] == "completed_with_errors"
    assert (state["attempted_pages"], state["completed_pages"], state["failed_pages"]) == (2, 1, 1)
    assert state["cases"][0]["pages"][0]["error"] == "fixture error"
    assert state["cases"][0]["pages"][1]["original_pdf_page"] == 74
    assert (tmp_path / "outputs/parser_runs/fixture/page-jobs/page_0001/run-status.json").exists()
