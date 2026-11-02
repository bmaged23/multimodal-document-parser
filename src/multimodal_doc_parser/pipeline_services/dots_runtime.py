"""Own a GPU-only dots server for one worker job, then release its GPU memory."""

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from multimodal_doc_parser.parser_adapters.defaults import DOTS_MODEL, DOTS_RUNTIME
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings


def verified_models(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    models = []
    for artifact in DOTS_RUNTIME["artifacts"]:
        if artifact["kind"] != "model":
            continue
        target = directory / artifact["filename"]
        valid = target.exists() and target.stat().st_size == artifact["size_bytes"]
        if valid:
            with target.open("rb") as handle:
                valid = hashlib.file_digest(handle, "sha256").hexdigest() == artifact["sha256"]
        if not valid:
            temporary = target.with_suffix(".download")
            try:
                with urllib.request.urlopen(artifact["url"], timeout=120) as source:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                with temporary.open("rb") as handle:
                    checksum = hashlib.file_digest(handle, "sha256").hexdigest()
                if (
                    temporary.stat().st_size != artifact["size_bytes"]
                    or checksum != artifact["sha256"]
                ):
                    raise ValueError("Downloaded dots model failed its pinned size/SHA-256 check")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        models.append(target)
    return models


@contextlib.contextmanager
def dots_server(settings: ServiceSettings, evidence: Path) -> Iterator[Path | None]:
    """External servers are deployment-owned; embedded servers live inside the GPU lease."""
    if settings.dots_binary is None:
        yield None
        return
    binary = settings.dots_binary
    probe = subprocess.run(
        [str(binary), "--list-devices"], capture_output=True, text=True, timeout=30, check=True
    )
    if not re.search(r"CUDA\d+:", probe.stdout + probe.stderr):
        raise ValueError("Quantized dots requires an NVIDIA CUDA GPU; CPU fallback is disabled")
    language, vision = verified_models(settings.dots_models)
    command = [
        str(binary),
        "--model",
        str(language),
        "--mmproj",
        str(vision),
        "--alias",
        DOTS_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8010",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "99",
        "--parallel",
        "1",
        "--threads",
        "6",
        "--threads-batch",
        "6",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--flash-attn",
        "off",
    ]
    environment = {**os.environ, "GGML_CUDA_PDL": "0"}
    metadata = evidence / "dots-runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "model_version": DOTS_RUNTIME["model_version"],
                "runtime": "llama.cpp-b11960-CUDA12.6",
                "execution_device": "gpu",
                "quantization": "Q4_K_M language / Q8_0 projector",
                "parallel_requests": 1,
                "context_size": 8192,
                "flash_attention": False,
                "server_command": command,
            },
            indent=2,
        )
        + "\n"
    )
    with (evidence / "dots-server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + settings.dots_startup_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("dots server exited; inspect retained dots-server.log")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8010/health", timeout=2
                    ) as response:
                        ready = response.status == 200
                except (urllib.error.URLError, TimeoutError, OSError):
                    ready = False
                if ready:
                    yield metadata
                    return
                time.sleep(0.2)
            raise TimeoutError("dots server startup timed out; inspect dots-server.log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
