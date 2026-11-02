"""Cloud credentials must stay out of images, model prompts and saved responses."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_doc_parser.visual_enrichment.qwen_cloud import NoRedirect, QwenCloudClient


def test_cloud_request_and_native_response_retention(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture-private-key")
    captured = []

    def open_request(request, timeout):
        captured.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "qwen3.8-flash",
                    "id": "fixture",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"kind":"image"}'}}
                    ],
                    "usage": {"completion_tokens": 8},
                }
            ).encode()
        )

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *a: SimpleNamespace(open=open_request)
    )
    client = QwenCloudClient("qwen3.8-flash", dotenv=tmp_path / "absent")
    response = client.api(
        "",
        "/api/generate",
        {
            "prompt": "Describe the image",
            "format": {"type": "object"},
            "images": ["fixture-base64"],
            "options": {"num_predict": 100, "temperature": 0},
        },
    )
    request = captured[0]
    assert request.get_header("Authorization") == "Bearer fixture-private-key"
    assert b"fixture-private-key" not in request.data
    body = json.loads(request.data)
    assert body["enable_thinking"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"][1]["image_url"]["url"].endswith("fixture-base64")
    assert response["response"] == '{"kind":"image"}'
    assert response["native_response"]["id"] == "fixture"
    assert "fixture-private-key" not in json.dumps(response)


def test_missing_key_and_untrusted_endpoint_fail_before_inference(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="Configure DASHSCOPE_API_KEY"):
        QwenCloudClient("qwen3.8-flash", dotenv=tmp_path / "absent")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture-private-key")
    with pytest.raises(ValueError, match="official regional"):
        QwenCloudClient("qwen3.8-flash", base_url="https://example.com/compatible-mode/v1")
    with pytest.raises(ValueError, match="redirects"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")
