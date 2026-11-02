"""Build pinned llama.cpp for the GTX 1660 Ti using an isolated CUDA 12.6 toolchain."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

from setup_dots_mocr_runtime import digest

ROOT = Path(__file__).resolve().parents[1]


def download(artifact: dict, destination: Path) -> None:
    if destination.exists() and digest(destination) == artifact["sha256"]:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}", flush=True)
    with urlopen(artifact["url"], timeout=90) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if digest(temporary) != artifact["sha256"]:
        raise ValueError(f"Checksum mismatch: {destination.name}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    backend = ROOT / "backends/dots_mocr_runtime"
    lock = json.loads((backend / "native-build-lock.json").read_text())
    toolchain_lock = backend / "toolchain-linux-64.lock"
    if digest(toolchain_lock) != lock["toolchain_lock_sha256"]:
        raise ValueError("Toolchain lock changed; regenerate its pinned hash")
    cache = ROOT / ".cache/dots_mocr"
    build = cache / "cuda126-build"
    build.mkdir(parents=True, exist_ok=True)
    for artifact, directory in [(lock["bootstrap"], "bootstrap"), (lock["source"], "source")]:
        archive_path = build / artifact["filename"]
        download(artifact, archive_path)
        if not (build / directory).exists():
            with tarfile.open(archive_path) as archive:
                archive.extractall(build / directory, filter="data")
    mamba = build / "bootstrap/bin/micromamba"
    prefix = build / "toolchain"
    environment = os.environ.copy()
    environment["MAMBA_ROOT_PREFIX"] = str(build / "mamba-root")
    if not (prefix / "conda-meta").exists():
        subprocess.run(
            [
                str(mamba),
                "create",
                "-y",
                "--no-rc",
                "-p",
                str(prefix),
                "--file",
                str(toolchain_lock),
            ],
            env=environment,
            check=True,
        )
    else:
        expected = {
            line for line in toolchain_lock.read_text().splitlines() if line.startswith("https://")
        }
        records = [json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")]
        actual = {record["url"] + "#" + record["md5"] for record in records}
        if actual != expected:
            raise ValueError("Existing native toolchain differs from the pinned package lock")
    source = build / "source/llama.cpp-b11960"
    runtime = cache / lock["runtime"]
    run = [str(mamba), "run", "-p", str(prefix)]
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(runtime / "build"),
        "-G",
        "Ninja",
        *lock["cmake_options"],
    ]
    subprocess.run(run + configure, env=environment, check=True)
    subprocess.run(
        run
        + [
            "cmake",
            "--build",
            str(runtime / "build"),
            "--target",
            "llama-server",
            "-j",
            str(args.jobs),
        ],
        env=environment,
        check=True,
    )
    metadata = json.loads((backend / "runtime-lock.json").read_text())
    metadata.update(
        runtime=lock["runtime"],
        native_build=lock,
        lock_sha256=digest(backend / "runtime-lock.json"),
        cache_directory=str(cache),
        artifacts=[artifact for artifact in metadata["artifacts"] if artifact["kind"] == "model"],
    )
    (cache / "installed-runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Runtime ready: {runtime}", flush=True)


if __name__ == "__main__":
    main()
