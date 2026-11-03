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


def test_cache_requires_same_model_digest_images_and_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "fixture-private-key")
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": "{}"},
                }
            ).encode()
        )

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *a: SimpleNamespace(open=open_request)
    )
    client = OllamaCloudClient("gemma4:31b", cache_directory=tmp_path / "cache")
    client.cache_namespace = "checkpoint-one"
    payload = {"model": "gemma4:31b", "messages": [{"content": "target", "images": ["one"]}]}
    assert client.request("/api/chat", payload) == client.request("/api/chat", payload)
    assert len(calls) == 1
    changed = {**payload, "messages": [{"content": "target", "images": ["two"]}]}
    client.request("/api/chat", changed)
    assert len(calls) == 2
    changed_prompt = {**payload, "messages": [{"content": "another target", "images": ["one"]}]}
    client.request("/api/chat", changed_prompt)
    assert len(calls) == 3
    client.cache_namespace = "checkpoint-two"
    client.request("/api/chat", payload)
    assert len(calls) == 4
    assert all("fixture-private-key" not in p.read_text() for p in (tmp_path / "cache").glob("*"))
