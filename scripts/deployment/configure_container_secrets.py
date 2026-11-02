"""Create local Compose secret files without printing or replacing existing values."""

import getpass
import os
import secrets
from pathlib import Path


def main() -> None:
    root = Path(".cache/service_secrets")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    for name in ["pipeline_api_key", "ollama_api_key"]:
        target = root / name
        if target.exists():
            print(f"Preserving existing {name}")
            continue
        value = (
            secrets.token_urlsafe(32)
            if name == "pipeline_api_key"
            else getpass.getpass("Ollama Cloud key (Enter for OCR only): ").strip()
        )
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value + "\n")
        # Private parent directory protects host access; Compose mounts files for UID 1000.
        os.chmod(target, 0o644)
    print("Secret files prepared; no credentials printed.")


if __name__ == "__main__":
    main()
