"""Create ignored local service credentials without printing or rotating existing keys."""

import os
import secrets
from pathlib import Path

from multimodal_doc_parser.visual_enrichment.qwen_cloud import configuration


def main() -> None:
    settings = configuration(names={"PIPELINE_API_KEY", "OLLAMA_API_KEY"})
    key = settings.get("PIPELINE_API_KEY") or secrets.token_urlsafe(32)
    if len(key) < 24:
        raise ValueError("Existing PIPELINE_API_KEY is too short")
    dotenv = Path(".env")
    if not settings.get("PIPELINE_API_KEY"):
        lines = (
            [
                line
                for line in dotenv.read_text().splitlines()
                if line.partition("=")[0].strip() != "PIPELINE_API_KEY"
            ]
            if dotenv.exists()
            else []
        )
        lines.append("PIPELINE_API_KEY=" + key)
        temporary = dotenv.with_name(".env.service-key.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("\n".join(lines) + "\n")
        temporary.replace(dotenv)
    root = Path(".cache/service_secrets")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, value in [
        ("pipeline_api_key", key),
        ("ollama_api_key", settings.get("OLLAMA_API_KEY")),
    ]:
        if not value:
            continue
        path = root / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(value + "\n")
    print(
        "Service authentication configured; existing Ollama credential retained. No keys printed."
    )


if __name__ == "__main__":
    main()
