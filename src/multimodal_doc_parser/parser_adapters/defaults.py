"""Pinned production parser identity; historical evaluation profiles stay separate."""

import json
from importlib.resources import files
from typing import Any

DEFAULT_BACKEND = "dots-mocr"
DOTS_RUNTIME: dict[str, Any] = json.loads(
    files("multimodal_doc_parser.parser_adapters.prompts")
    .joinpath("dots_runtime_lock.json")
    .read_text()
)
DOTS_MODEL = str(DOTS_RUNTIME["server_model_alias"])
DOTS_VERSION = str(DOTS_RUNTIME["model_version"])
DOTS_MAX_TOKENS = 6144
