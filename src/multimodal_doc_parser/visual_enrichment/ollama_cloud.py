"""Ollama Cloud chat transport; retain native output and validate JSON locally."""

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from multimodal_doc_parser.visual_enrichment.qwen_cloud import NoRedirect, configuration


class OllamaCloudClient:
    def __init__(self, model: str, dotenv: Path = Path(".env")) -> None:
        self.key = configuration(dotenv, {"OLLAMA_API_KEY"}).get("OLLAMA_API_KEY")
        if not self.key:
            raise ValueError("Configure OLLAMA_API_KEY in the environment or ignored project .env")
        self.model = model
        self.base_url = "https://ollama.com"

    def request(
        self, route: str, data: dict[str, Any] | None = None, timeout: float = 300
    ) -> dict[str, Any]:
        if route not in {"/api/tags", "/api/chat"}:
            raise ValueError("Unsupported Ollama Cloud route")
        request = urllib.request.Request(
            self.base_url + route,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            result: dict[str, Any] = json.load(response)
        if result.get("error"):
            raise ValueError("Ollama Cloud returned an API error")
        return result

    def api(
        self, base_url: str, route: str, data: dict[str, Any] | None = None, timeout: float = 300
    ) -> dict[str, Any]:
        if route != "/api/generate" or data is None:
            raise ValueError("Ollama Cloud adapter only accepts generation requests")
        prompt = (
            data["prompt"] + "\nOutput ONE JSON object only, without Markdown fences or prose. "
            "Match this JSON schema:\n" + json.dumps(data["format"])
        )
        native = self.request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": data["images"]}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": data["options"]["temperature"],
                    "num_predict": data["options"]["num_predict"],
                },
            },
            timeout,
        )
        content = native["message"]["content"]
        # Unwrap only a complete JSON code fence; never repair or extract JSON from prose.
        fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", content.strip())
        return {
            "response": fence[1] if fence else content,
            "response_envelope": "single_json_code_fence" if fence else "plain",
            "native_response": native,
            "done": native.get("done"),
            "done_reason": native.get("done_reason"),
            "eval_count": native.get("eval_count"),
            "total_duration": native.get("total_duration"),
            "returned_model": native.get("model"),
            "execution": "ollama-cloud",
        }
