"""Download manifest-listed PDF sources locally, preserving reproducible checksums."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def download_document(url: str, destination: Path, expected_hash: str | None) -> str:
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Existing file checksum mismatch: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "MultimodalDocParser-Evaluation/0.1"})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            hasher = hashlib.sha256()
            size = 0
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 300 * 1024 * 1024:
                        raise ValueError("Download exceeds the 300 MiB safety limit")
                    target.write(chunk)
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        with temporary.open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                raise ValueError(f"Source did not return a PDF: {url}")
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Downloaded file checksum mismatch: {destination}")
        temporary.rename(destination)
        return digest
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/corpus-manifest.json"))
    parser.add_argument("--document-id", action="append", help="Download only selected IDs")
    parser.add_argument(
        "--output-directory", type=Path, default=Path("data/input_documents/evaluation_corpus")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if args.document_id:
        unknown = set(args.document_id) - {item["id"] for item in documents}
        if unknown:
            parser.error(f"Unknown document IDs: {sorted(unknown)}")
        documents = [item for item in documents if item["id"] in args.document_id]
    for document in documents:
        destination = args.output_directory / document["filename"]
        print(f"Downloading/checking {document['id']}...", flush=True)
        digest = download_document(document["download_url"], destination, document.get("sha256"))
        print(
            json.dumps({"id": document["id"], "path": str(destination), "sha256": digest}),
            flush=True,
        )


if __name__ == "__main__":
    main()
