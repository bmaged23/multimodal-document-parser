"""Download the pinned official Unlimited-OCR 2 snapshot, verifying LFS hashes."""

import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"
REPOSITORY = "baidu/Unlimited-OCR-2"


def main() -> None:
    cache = ROOT / ".cache/unlimited_ocr_2"
    source = ROOT / "backends/unlimited_ocr_2_runtime/model-lock.json"
    metadata = json.loads(source.read_text())
    if metadata["sha"] != REVISION:
        raise ValueError("Source review revision differs from pinned snapshot")
    target = cache / REVISION
    target.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in metadata["siblings"]
        if "/" not in f["rfilename"]
        and f["rfilename"].endswith((".py", ".json", ".safetensors", ".md"))
    ]

    def download(item: dict) -> None:
        name = item["rfilename"]
        destination = target / name
        expected = item.get("lfs", {}).get("sha256")
        if destination.exists() and destination.stat().st_size == item["size"]:
            if expected:
                with destination.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest == expected:
                    return
            else:
                blob = b"blob " + str(item["size"]).encode() + b"\0" + destination.read_bytes()
                if hashlib.sha1(blob).hexdigest() == item["blobId"]:
                    return
        temporary = destination.with_suffix(destination.suffix + ".partial")
        print(f"Downloading {name} ({item['size']:,} bytes)", flush=True)
        with (
            urllib.request.urlopen(
                f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", timeout=120
            ) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, 8 * 1024 * 1024)
        if temporary.stat().st_size != item["size"]:
            raise ValueError(f"Incorrect byte size: {name}")
        with temporary.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if expected and expected != actual:
            raise ValueError(f"SHA-256 mismatch: {name}")
        if not expected:
            blob = b"blob " + str(item["size"]).encode() + b"\0" + temporary.read_bytes()
            if hashlib.sha1(blob).hexdigest() != item["blobId"]:
                raise ValueError(f"Git blob mismatch: {name}")
        temporary.replace(destination)
        print(f"Verified {name}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": []}
    for item in files:
        path = target / item["rfilename"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest["files"].append(
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    (cache / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Official snapshot ready: {target}", flush=True)


if __name__ == "__main__":
    main()
