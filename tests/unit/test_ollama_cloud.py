"""Hosted vision calls retain native output and never pretend to enforce server schemas."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

from multimodal_doc_parser.visual_enrichment.ollama_cloud import OllamaCloudClient


def test_cloud_uses_prompt_schema_and_retains_fenced_native_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "fixture-private-key")
    captured = []

    def open_request(request, timeout):
        captured.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "gemma4:31b",
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": '```json\n{"kind":"image"}\n```'},
                }
            ).encode()
        )

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *a: SimpleNamespace(open=open_request)
    )
    client = OllamaCloudClient("gemma4:31b", dotenv=tmp_path / "absent")
    result = client.api(
        "",
        "/api/generate",
        {
            "prompt": "Inspect target",
            "format": {"type": "object"},
            "images": ["fixture-image"],
            "options": {"temperature": 0, "num_predict": 100},
        },
    )
    body = json.loads(captured[0].data)
    assert "format" not in body
    assert "JSON schema" in body["messages"][0]["content"]
    assert body["messages"][0]["images"] == ["fixture-image"]
    assert "fixture-private-key" not in json.dumps(body)
    assert result["response"] == '{"kind":"image"}'
    assert result["response_envelope"] == "single_json_code_fence"
    assert result["native_response"]["message"]["content"].startswith("```json")
