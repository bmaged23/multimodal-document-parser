"""Serve quantized dots.mocr on GPU through the isolated native runtime."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def server_command(
    binary: Path,
    cache: Path,
    model: dict[str, Any],
    *,
    port: int,
    context_size: int,
) -> list[str]:
    language = "dots.mocr.Q4_K_M.gguf"
    vision = "dots.mocr.mmproj-q8_0.gguf"
    command = [
        str(binary),
        "--model",
        str(cache / language),
        "--mmproj",
        str(cache / vision),
        "--alias",
        model["server_model_alias"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
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
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--context-size", type=int, default=8192)
    args = parser.parse_args()
    cache = ROOT / ".cache/dots_mocr"
    installed_path = cache / "installed-runtime.json"
    if not installed_path.exists():
        parser.error("Set up and build the native dots.mocr runtime first")
    metadata = json.loads(installed_path.read_text())
    runtime = cache / metadata["runtime"]
    binaries = list(runtime.rglob("llama-server"))
    if len(binaries) != 1:
        parser.error("Expected exactly one llama-server in the pinned runtime")
    libraries = {str(path.parent) for path in runtime.rglob("*.so*")}
    environment = os.environ.copy()
    environment["GGML_CUDA_PDL"] = "0"
    environment["LD_LIBRARY_PATH"] = (
        ":".join(sorted(libraries)) + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    command = server_command(
        binaries[0],
        cache,
        metadata,
        port=args.port,
        context_size=args.context_size,
    )
    launch_path = cache / "server-launch.json"
    launch_path.write_text(
        json.dumps(
            {
                **metadata,
                "model_profile": "gpu-q4",
                "server_command": command,
                "context_size": args.context_size,
                "gpu_layers": 99,
                "vision_gpu_offload": True,
                "execution_device": "gpu",
                "parallel_requests": 1,
                "flash_attention": False,
                "GGML_CUDA_PDL": "0",
                "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Starting {metadata['runtime']} profile=gpu-q4", flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
