"""Qwen Cloud vision transport with scoped credentials and retained native responses."""

import json
import os
import shlex
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"
REGIONS = {
    "ap-southeast-1",
    "cn-beijing",
    "eu-central-1",
    "cn-hongkong",
    "ap-northeast-1",
    "us-east-1",
}


def configuration(dotenv: Path = Path(".env"), names: set[str] | None = None) -> dict[str, str]:
    """Read only the named settings; never execute or log dotenv content."""
    allowed = names if names is not None else {"DASHSCOPE_API_KEY", "QWEN_API_KEY", "QWEN_BASE_URL"}
    values = {}
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            name, separator, value = line.removeprefix("export ").partition("=")
            if separator and name.strip() in allowed:
                tokens = shlex.split(value, comments=True)
                if len(tokens) != 1:
                    raise ValueError("Invalid Qwen setting in .env; use one quoted or plain value")
                values[name.strip()] = tokens[0]
    values.update({name: os.environ[name] for name in allowed if os.environ.get(name)})
    return values


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise ValueError("Cloud API redirects are disabled to protect credentials")


class QwenCloudClient:
    def __init__(
        self, model: str, base_url: str | None = None, dotenv: Path = Path(".env")
    ) -> None:
        settings = configuration(dotenv)
        self.base_url = (base_url or settings.get("QWEN_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        url = urlsplit(self.base_url)
        hosts = {region + ".maas.qwencloudapi.com" for region in REGIONS}
        if (
            url.scheme != "https"
            or url.netloc not in hosts
            or url.path != "/compatible-mode/v1"
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Use an official regional Qwen Cloud HTTPS compatible-mode/v1 endpoint"
            )
        self.key = settings.get("DASHSCOPE_API_KEY") or settings.get("QWEN_API_KEY")
        if not self.key:
            raise ValueError(
                "Configure DASHSCOPE_API_KEY in the environment or ignored project .env; "
                "the key and QWEN_BASE_URL must belong to the same region"
            )
        self.model = model

    def api(
        self, base_url: str, route: str, data: dict[str, Any] | None = None, timeout: float = 300
    ) -> dict[str, Any]:
        if route != "/api/generate" or data is None:
            raise ValueError("Qwen Cloud transport only accepts generation requests")
        schema = data["format"]
        text = (
            data["prompt"]
            + "\nReturn only a JSON object matching this schema:\n"
            + json.dumps(schema)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}
            for image in data["images"]
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "max_tokens": data["options"]["num_predict"],
            "temperature": data["options"]["temperature"],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            native: dict[str, Any] = json.load(response)
        if native.get("error"):
            raise ValueError("Qwen Cloud returned an API error")
        choice = native["choices"][0]
        return {
            "response": choice["message"]["content"],
            "done": True,
            "done_reason": choice["finish_reason"],
            "native_response": native,
            "eval_count": native.get("usage", {}).get("completion_tokens"),
            "execution": "qwen-cloud",
            "requested_model": self.model,
            "returned_model": native.get("model"),
        }
