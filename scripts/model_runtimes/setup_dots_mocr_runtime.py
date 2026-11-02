"""Download checksum-pinned dots.mocr GGUF weights and an isolated llama.cpp runtime."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prebuilt",
        action="store_true",
        help="Use the CUDA 12.8 prebuilt runtime (incompatible with driver 560.76)",
    )
    args = parser.parse_args()
    lock_path = ROOT / "backends/dots_mocr_runtime/runtime-lock.json"
    lock = json.loads(lock_path.read_text())
    cache = ROOT / ".cache/dots_mocr"
    cache.mkdir(parents=True, exist_ok=True)
    artifacts = [item for item in lock["artifacts"] if args.prebuilt or item["kind"] == "model"]
    for artifact in artifacts:
        destination = cache / artifact["filename"]
        if not destination.exists() or digest(destination) != artifact["sha256"]:
            temporary = destination.with_suffix(destination.suffix + ".part")
            print(
                f"Downloading {destination.name} ({artifact['size_bytes'] / 1e6:.0f} MB)",
                flush=True,
            )
            with urlopen(artifact["url"], timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if (
                temporary.stat().st_size != artifact["size_bytes"]
                or digest(temporary) != artifact["sha256"]
            ):
                raise ValueError(f"Artifact integrity check failed: {destination.name}")
            temporary.replace(destination)
        print(f"Verified {destination.name}", flush=True)
    runtime = cache / lock["runtime"]
    if args.prebuilt and not runtime.exists():
        staging = Path(tempfile.mkdtemp(prefix=".runtime-", dir=cache))
        try:
            for artifact in lock["artifacts"]:
                if artifact["kind"] == "runtime_archive":
                    with tarfile.open(cache / artifact["filename"]) as archive:
                        archive.extractall(staging, filter="data")
            staging.replace(runtime)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if not args.prebuilt:
        print(
            "Model files ready. Run scripts/model_runtimes/build_dots_mocr_runtime.py "
            "for the local GPU runtime.",
            flush=True,
        )
        return
    installed = {**lock, "lock_sha256": digest(lock_path), "cache_directory": str(cache)}
    (cache / "installed-runtime.json").write_text(json.dumps(installed, indent=2) + "\n")
    print(f"Runtime ready: {runtime}", flush=True)


if __name__ == "__main__":
    main()
