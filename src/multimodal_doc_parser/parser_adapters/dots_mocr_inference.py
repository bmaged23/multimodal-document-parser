"""Bounded, sequential dots.mocr requests through an OpenAI-compatible server."""

import base64
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import decode_layout

# The upstream MIT notice is retained beside these packaged prompt resources.
PROMPT_RESOURCES = files("multimodal_doc_parser.parser_adapters").joinpath("prompts")
LAYOUT_PROMPT = PROMPT_RESOURCES.joinpath("dots_mocr_layout_all_en.txt").read_text()
PROMPT_SOURCE = json.loads(PROMPT_RESOURCES.joinpath("dots_mocr_prompt_source.json").read_text())


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(url: str, timeout: float, payload: dict[str, Any] | None = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("DOTS_MOCR_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def input_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Use multiples of 28, as required by the model's visual patch grid."""
    if max_pixels < 28 * 28:
        raise ValueError("max_pixels must be at least 784")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Image aspect ratio exceeds dots.mocr limits")
    scaled_width, scaled_height = max(28, round(width / 28) * 28), max(28, round(height / 28) * 28)
    if scaled_width * scaled_height > max_pixels:
        factor = math.sqrt(width * height / max_pixels)
        scaled_width = max(28, math.floor(width / factor / 28) * 28)
        scaled_height = max(28, math.floor(height / factor / 28) * 28)
    if scaled_width * scaled_height > max_pixels:
        raise ValueError("Image cannot fit the requested pixel budget")
    return scaled_width, scaled_height


def run_dots_mocr_inference(
    document: Document,
    bundle: Path,
    raw: Path,
    *,
    base_url: str,
    model: str,
    model_version: str,
    server_kind: str,
    timeout_seconds: float,
    max_tokens: int,
    max_pixels: int,
    runtime_metadata: dict[str, Any] | None = None,
) -> None:
    if hashlib.sha256(LAYOUT_PROMPT.encode()).hexdigest() != PROMPT_SOURCE["prompt_sha256"]:
        raise ValueError("Packaged dots.mocr prompt differs from its pinned source")
    if timeout_seconds <= 0 or max_tokens < 1 or max_pixels < 784:
        raise ValueError("Timeout, token, and pixel limits must be positive")
    if server_kind not in {"llama-cpp", "vllm"}:
        raise ValueError("Unsupported dots.mocr server kind")
    raw.mkdir(parents=True)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "initializing",
        "document_id": document.document_id,
        "backend": "dots.mocr",
        "model": model,
        "model_version": model_version,
        "server_kind": server_kind,
        "runtime_metadata": runtime_metadata,
        "prompt_version": "upstream-layout-all-en@" + PROMPT_SOURCE["source_commit"],
        "prompt_sha256": hashlib.sha256(LAYOUT_PROMPT.encode()).hexdigest(),
        "max_tokens": max_tokens,
        "max_pixels": max_pixels,
        "timeout_seconds": timeout_seconds,
        "temperature": 0.0,
        "top_p": 1.0,
        "completed_pages": 0,
        "total_pages": len(document.pages),
        "pages": [],
        "memory_measurement": (
            "Local GPU 0 total memory sampled once per second when available; includes other "
            "processes, excludes server startup, not allocator peak. Remote GPUs are not sampled."
        ),
        "server_initialization_included": False,
        "error": None,
    }

    def checkpoint(state: str) -> None:
        manifest.update(
            status=state,
            total_seconds=time.perf_counter() - started,
            peak_sampled_device_memory_mib=max(memory_samples) if memory_samples else None,
        )
        write_json(raw / "inference-manifest.json", manifest)

    stop = threading.Event()
    memory_samples: list[int] = []

    def sample_local_gpu() -> None:
        if urlparse(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            return
        while not stop.is_set():
            try:
                sample = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                )
                memory_samples.append(int(sample.stdout.splitlines()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            stop.wait(1)

    sampler = threading.Thread(target=sample_local_gpu, daemon=True)
    sampler.start()
    checkpoint("initializing")
    try:
        available = request_json(base_url.rstrip("/") + "/models", timeout_seconds)
        if model not in [item["id"] for item in available["data"]]:
            raise ValueError(f"Server does not advertise the requested model {model!r}")
        for page in document.pages:
            manifest["current_page"] = page.page_number
            manifest["page_started_at_unix"] = time.time()
            checkpoint("running")
            print(f"Starting page {page.page_number}/{len(document.pages)}", flush=True)
            tick = time.perf_counter()
            with Image.open(bundle / page.image.path) as source:
                width, height = input_size(source.width, source.height, max_pixels)
                with source.convert("RGB") as rgb:
                    with rgb.resize((width, height), Image.Resampling.LANCZOS) as image:
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
            image_path = raw / "input-images" / f"page_{page.page_number:04d}.png"
            image_path.parent.mkdir(exist_ok=True)
            image_path.write_bytes(image_bytes)
            prompt = LAYOUT_PROMPT.format(width=width, height=height)
            if server_kind == "vllm":
                prompt = "<|img|><|imgpad|><|endofimg|>" + prompt
            response = request_json(
                base_url.rstrip("/") + "/chat/completions",
                timeout_seconds,
                {
                    "model": model,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                        + base64.b64encode(image_bytes).decode("ascii")
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                },
            )
            # Always retain the complete server response, including truncated or malformed output.
            write_json(raw / f"response_{page.page_number:04d}.json", response)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") != "stop":
                raise ValueError(
                    f"Page {page.page_number}: incomplete generation "
                    f"({choice.get('finish_reason')})"
                )
            decode_layout(content)
            elapsed = time.perf_counter() - tick
            write_json(
                raw / f"page_{page.page_number:04d}.json",
                {
                    "format_version": "dots-mocr-page-v1",
                    "page_number": page.page_number,
                    "source_width": page.width_pixels,
                    "source_height": page.height_pixels,
                    "input_width": width,
                    "input_height": height,
                    "coordinate_space": "input_pixels",
                    "input_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "finish_reason": choice["finish_reason"],
                    "content": content,
                    "usage": response.get("usage"),
                },
            )
            manifest["pages"].append(
                {
                    "page_number": page.page_number,
                    "elapsed_seconds": elapsed,
                    "input_width": width,
                    "input_height": height,
                    "usage": response.get("usage"),
                }
            )
            manifest["completed_pages"] += 1
            checkpoint("running")
            print(
                f"Completed page {page.page_number}/{len(document.pages)} in {elapsed:.1f}s",
                flush=True,
            )
        manifest["current_page"] = None
        checkpoint("completed")
    except Exception as exc:
        manifest["error"] = str(exc)
        checkpoint("failed")
        raise
    finally:
        stop.set()
        sampler.join(timeout=4)
