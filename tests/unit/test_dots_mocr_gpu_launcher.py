"""The supported dots.mocr launcher must use the retained quantized GPU models."""

import runpy
from pathlib import Path


def test_gpu_launcher_selects_retained_models_and_enables_offload():
    build = runpy.run_path("scripts/serve_dots_mocr.py")["server_command"]
    command = build(
        Path("llama-server"),
        Path("cache"),
        {"server_model_alias": "q4"},
        port=8010,
        context_size=8192,
    )
    assert command[command.index("--n-gpu-layers") + 1] == "99"
    assert command[command.index("--model") + 1] == "cache/dots.mocr.Q4_K_M.gguf"
    assert command[command.index("--mmproj") + 1] == "cache/dots.mocr.mmproj-q8_0.gguf"
    assert "--no-mmproj-offload" not in command
    assert not any("BF16" in part or "bf16" in part for part in command)
