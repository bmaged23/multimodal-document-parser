"""Production selection, pinned identity and managed GPU process lifecycle."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_doc_parser.parser_adapters.defaults import DOTS_RUNTIME, DOTS_VERSION
from multimodal_doc_parser.pipeline_services import dots_runtime
from multimodal_doc_parser.pipeline_services.contracts import ParsingJob, PipelineJob
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings


def test_api_defaults_and_packaged_model_lock():
    assert ParsingJob(source_artifact_id="a" * 32).backend == "dots-mocr"
    assert PipelineJob(source_artifact_id="a" * 32).backend == "dots-mocr"
    assert DOTS_RUNTIME == json.loads(
        Path("backends/dots_mocr_runtime/runtime-lock.json").read_text()
    )


def test_cli_default_passes_the_pinned_version(monkeypatch, tmp_path):
    from multimodal_doc_parser import command_line

    captured = {}

    def parse(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(command_line, "parse_prepared_document", parse)
    monkeypatch.setattr("sys.argv", ["document-parser", "parse", str(tmp_path)])
    command_line.main()
    assert captured["backend"] == "dots-mocr"
    assert captured["model_version"] == DOTS_VERSION
    assert captured["max_tokens"] == 6144


def test_managed_server_refuses_cpu_fallback(monkeypatch, tmp_path):
    settings = ServiceSettings(storage=tmp_path, api_key="fixture", dots_binary=Path("/fake"))
    monkeypatch.setattr(
        dots_runtime.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="Available devices:", stderr=""),
    )
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        with dots_runtime.dots_server(settings, tmp_path):
            pytest.fail("CPU fallback must never enter parsing")


def test_managed_server_stops_and_preserves_original_parsing_exception(monkeypatch, tmp_path):
    settings = ServiceSettings(storage=tmp_path, api_key="fixture", dots_binary=Path("/fake"))
    monkeypatch.setattr(
        dots_runtime.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="CUDA0: Fixture GPU", stderr=""),
    )
    monkeypatch.setattr(
        dots_runtime, "verified_models", lambda _: [tmp_path / "language", tmp_path / "vision"]
    )
    state = {"terminated": False}

    class Process:
        def poll(self):
            return None if not state["terminated"] else 0

        def terminate(self):
            state["terminated"] = True

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(dots_runtime.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(
        dots_runtime.urllib.request,
        "urlopen",
        lambda *a, **k: contextlib.nullcontext(SimpleNamespace(status=200)),
    )
    with pytest.raises(OSError, match="original extraction error"):
        with dots_runtime.dots_server(settings, tmp_path) as metadata:
            assert json.loads(metadata.read_text())["execution_device"] == "gpu"
            raise OSError("original extraction error")
    assert state["terminated"]
