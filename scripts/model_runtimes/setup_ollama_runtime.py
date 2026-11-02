"""Install a checksummed portable Ollama release in the ignored project cache."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parents[2]
VERSION = "v0.40.13"
DIGEST = "c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6"
HOST = "http://127.0.0.1:11435"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Explicit local model to pull; omitted for server-only setup"
    )
    args = parser.parse_args()
    cache = ROOT / ".cache/ollama"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"ollama-{VERSION}-linux-amd64.tar.zst"
    binary = cache / VERSION / "bin/ollama"
    if not binary.exists():
        if not archive.exists():
            url = f"https://github.com/ollama/ollama/releases/download/{VERSION}/ollama-linux-amd64.tar.zst"
            temporary = archive.with_suffix(".partial")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                copied = 0
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
                    copied += len(block)
                    if copied % (128 * 1024 * 1024) == 0:
                        print(f"Ollama download: {copied / 1024**2:.0f} MiB", flush=True)
            temporary.replace(archive)
        with archive.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != DIGEST:
                raise ValueError("Ollama release SHA-256 mismatch")
        destination = cache / VERSION
        destination.mkdir(exist_ok=True)
        with (
            archive.open("rb") as handle,
            zstandard.ZstdDecompressor().stream_reader(handle) as stream,
        ):
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                tar.extractall(destination, filter="data")
        (cache / "runtime.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "release_sha256": DIGEST,
                    "binary": str(binary),
                    "host": HOST,
                    "models_directory": str(cache / "models"),
                },
                indent=2,
            )
            + "\n"
        )
    environment = os.environ.copy()
    environment.update(
        OLLAMA_HOST="127.0.0.1:11435",
        OLLAMA_MODELS=str(cache / "models"),
        OLLAMA_NUM_PARALLEL="1",
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_CONTEXT_LENGTH="4096",
        OLLAMA_NO_CLOUD="1",
    )
    try:
        with urllib.request.urlopen(HOST + "/api/version", timeout=2) as response:
            print(response.read().decode(), flush=True)
    except OSError:
        with (ROOT / "outputs/ollama-server.log").open("a") as log:
            process = subprocess.Popen(
                [str(binary), "serve"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (cache / "server-pid.json").write_text(json.dumps({"pid": process.pid}))
        for _ in range(90):
            if process.poll() is not None:
                raise RuntimeError(
                    "Ollama server exited; inspect outputs/ollama-server.log"
                ) from None
            try:
                with urllib.request.urlopen(HOST + "/api/version", timeout=2):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Ollama startup timed out")
    if not args.model:
        print("Ollama server ready; no local model download requested", flush=True)
        return
    request = urllib.request.Request(
        HOST + "/api/pull",
        method="POST",
        data=json.dumps({"model": args.model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = ""
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            row = json.loads(line)
            if row.get("error"):
                raise RuntimeError(row["error"])
            message = row.get("status", "")
            bucket = row.get("completed", 0) // (256 * 1024 * 1024)
            key = f"{message}:{bucket}"
            if key != last:
                print(message, row.get("completed", ""), row.get("total", ""), flush=True)
                last = key
    with urllib.request.urlopen(HOST + "/api/tags") as response:
        tags = json.load(response)
    (cache / "downloaded-models.json").write_text(json.dumps(tags, indent=2) + "\n")
    print("Ollama and vision model ready", flush=True)


if __name__ == "__main__":
    main()
