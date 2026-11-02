"""Wait for staged downloads, run a bounded GPU probe, preserve diagnostic artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"
RUNTIME = ROOT / "backends/unlimited_ocr_2_runtime"
CASE = "unlimited_ocr_2_nf4_original_two_pages_01"
RUN = ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-original-two-pages-01"
STATE = ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-validation-01-supervisor.json"


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def archive() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    state = json.loads((RUN / "run-status.json").read_text())
    destination = (
        ROOT
        / "outputs/evaluation_archive/Unlimited-OCR-2-NF4-FP16-vision_output"
        / CASE
        / f"{stamp}_{state['status']}"
    )
    destination.mkdir(parents=True)
    shutil.copytree(RUN, destination / "run")
    reproduction = destination / "reproduction"
    reproduction.mkdir()
    for source in [
        RUNTIME / "pyproject.toml",
        RUNTIME / "uv.lock",
        RUNTIME / "model-lock.json",
        RUNTIME / "run_gpu_probe.py",
        ROOT / "scripts/visualization/build_unlimited_ocr_2_review.py",
        ROOT / "scripts/parser_evaluation/run_unlimited_ocr_2_validation.py",
        ROOT / "scripts/parser_evaluation/run_unlimited_ocr_2_evaluation_batch.py",
        ROOT / ".cache/unlimited_ocr_2/download-manifest.json",
    ]:
        shutil.copyfile(source, reproduction / source.name)
    write(
        destination / "archive-metadata.json",
        {
            "case_id": CASE,
            "model": "baidu/Unlimited-OCR-2",
            "revision": REVISION,
            "profile": "NF4 language / FP16 vision / PyTorch attention fallback",
            "status": state["status"],
            "completed_generations": state["completed_pages"],
            "canonical_normalization": False,
            "accuracy_scored": False,
            "notes": "Diagnostic predictions; source PDF private. Not canonical parser output.",
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    inventory = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            inventory.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    write(destination / "file-checksums.json", {"algorithm": "sha256", "files": inventory})
    registry_path = ROOT / "evaluation/preserved-runs.json"
    registry = json.loads(registry_path.read_text())
    entries = registry.setdefault("experimental_cases", [])
    if not any(entry["id"] == CASE for entry in entries):
        entries.append(
            {
                "id": CASE,
                "run_directory": str(RUN.relative_to(ROOT)),
                "model_name": "Unlimited-OCR-2-NF4-FP16-vision",
                "expected_evaluation_pages": state["total_pages"],
                "evaluation_pages": state.get("selected_pages"),
                "archive_directory": str(destination.relative_to(ROOT)),
                "notes": "Diagnostic probe; not normalized project Document JSON.",
            }
        )
        write(registry_path, registry)
    return destination


def main() -> None:
    compiler = ROOT / ".cache/dots_mocr/cuda126-build/toolchain/bin/x86_64-conda-linux-gnu-cc"
    if compiler.exists():
        os.environ.setdefault("CC", str(compiler))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/unlimited_ocr_2/triton"))
    if RUN.exists():
        raise ValueError("Probe output exists; use a new case for another run")
    state = {
        "status": "waiting_for_downloads",
        "supervisor_pid": os.getpid(),
        "run_directory": str(RUN.relative_to(ROOT)),
        "model_revision": REVISION,
    }
    manifest = ROOT / ".cache/unlimited_ocr_2/download-manifest.json"
    snapshot = manifest.parent / REVISION
    deadline = time.monotonic() + 7200
    while True:
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        state["downloaded_weight_bytes"] = sum(
            p.stat().st_size for p in snapshot.glob("model-*.safetensors*")
        )
        state["expected_weight_bytes"] = 6_688_305_576
        write(STATE, state)
        if (
            manifest.exists()
            and (RUNTIME / ".venv/lib/python3.12/site-packages/torch/__init__.py").exists()
        ):
            check = subprocess.run(
                [
                    str(RUNTIME / ".venv/bin/python"),
                    "-c",
                    "import torch,transformers,bitsandbytes; assert torch.cuda.is_available()",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if check.returncode == 0:
                break
            state["environment_check_error"] = check.stderr[-2000:]
        if time.monotonic() >= deadline:
            state["status"] = "setup_failed"
            write(STATE, state)
            raise TimeoutError("Download/environment stage did not finish within two hours")
        time.sleep(10)
    state["status"] = "verifying_downloads"
    write(STATE, state)
    # Recheck all pinned file hashes before executing the local reviewed snapshot.
    lock = json.loads((RUNTIME / "model-lock.json").read_text())
    downloaded = json.loads(manifest.read_text())
    if downloaded["revision"] != REVISION:
        raise ValueError("Unexpected download revision")
    for item in lock["siblings"]:
        path = snapshot / item["rfilename"]
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Size mismatch: {path.name}")
        if "lfs" in item:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["lfs"]["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path.name}")
        else:
            blob = b"blob " + str(item["size"]).encode() + b"\0" + path.read_bytes()
            if hashlib.sha1(blob).hexdigest() != item["blobId"]:
                raise ValueError(f"Git blob mismatch: {path.name}")
    state["status"] = "running_gpu_probe"
    write(STATE, state)
    with (ROOT / "outputs/unlimited-ocr-2-nf4-original-two-pages-01.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(RUNTIME / ".venv/bin/python"),
                "-u",
                str(RUNTIME / "run_gpu_probe.py"),
                "--output-directory",
                str(RUN),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        state["probe_pid"] = process.pid
        write(STATE, state)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            current = json.loads((RUN / "run-status.json").read_text())
            current.update(
                status="interrupted", error="GPU probe exceeded 20-minute wall-clock limit"
            )
            write(RUN / "run-status.json", current)
            code = -1
    state["probe_exit_code"] = code
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/visualization/build_unlimited_ocr_2_review.py"),
            str(RUN),
        ],
        check=True,
    )
    state["archive_directory"] = str(archive().relative_to(ROOT))
    state["status"] = json.loads((RUN / "run-status.json").read_text())["status"]
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    write(STATE, state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
