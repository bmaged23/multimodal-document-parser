"""Package the exact preserved dots GPU input images for a private Colab run."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = [
    ("original-multimodal-2-pages", "dots_mocr_q4_original_multimodal_2_pages_upstream_prompt"),
    ("nasa-first-20-pages", "dots_mocr_q4_nasa_first20_01"),
    ("historical-diary-5-pages", "dots_mocr_q4_diary_five_01"),
    ("author-mathematics-4-pages", "dots_mocr_q4_author_handwritten_mathematics_four_pages_01"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(destination: Path, selections: list[tuple[str, str]] | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    registry = json.loads((ROOT / "evaluation/preserved-runs.json").read_text())
    manifest = {
        "version": "1.0",
        "model_id": "dots-studio/dots.mocr",
        "model_revision": "e539fbb52280393adc081b289ec597430a0f9031",
        "cases": [],
        "files": {},
        "private_inputs_included": True,
        "notes": (
            "Private matched inputs; do not publish this ZIP. No baseline predictions are included."
        ),
    }
    for name, identifier in SELECTIONS if selections is None else selections:
        case = next(c for c in registry["cases"] if c["id"] == identifier)
        run = ROOT / case["run_directory"]
        document = json.loads((run / "document.json").read_text())
        selected = (
            json.loads((ROOT / case["selection_manifest"]).read_text())
            if case.get("selection_manifest")
            else {}
        )
        originals = selected.get("original_pdf_pages", case["evaluation_pages"])
        folder = destination / name
        folder.mkdir()
        shutil.copyfile(run / "source.pdf", folder / "source.pdf")
        if "sha256:" + digest(folder / "source.pdf") != document["document_id"]:
            raise ValueError("Preserved source identity mismatch")
        item = {
            "id": name,
            "baseline_case_id": identifier,
            "source_document_id": document["document_id"],
            "pages": [],
        }
        for page, original in zip(document["pages"], originals, strict=True):
            number = page["page_number"]
            raw = run / "raw/dots_mocr"
            candidate = raw / "input-images" / f"page_{number:04d}.png"
            if not candidate.exists():
                raw = run / f"page-jobs/page_{number:04d}/run/raw/dots_mocr"
                candidate = raw / "input-images/page_0001.png"
            if not candidate.is_file():
                raise ValueError(f"Exact sent input image missing: {identifier} page {number}")
            target = folder / f"page_{number:04d}.png"
            shutil.copyfile(candidate, target)
            raw_page = run / "raw/dots_mocr" / f"page_{number:04d}.json"
            if raw_page.exists():
                if json.loads(raw_page.read_text())["input_image_sha256"] != digest(target):
                    raise ValueError("Saved prediction/input hash mismatch")
            with Image.open(target) as image:
                width, height = image.size
            item["pages"].append(
                {
                    "page_number": number,
                    "original_pdf_page": original,
                    "input_image": str(target.relative_to(destination)),
                    "input_image_sha256": digest(target),
                    "source_width": page["width_pixels"],
                    "source_height": page["height_pixels"],
                    "input_width": width,
                    "input_height": height,
                }
            )
        manifest["cases"].append(item)
    shutil.copyfile(
        ROOT / "src/multimodal_doc_parser/parser_adapters/prompts/dots_mocr_layout_all_en.txt",
        destination / "layout-prompt.txt",
    )
    shutil.copyfile(
        ROOT / "scripts/parser_evaluation/run_dots_mocr_unquantized_gpu.py",
        destination / "run_gpu_test.py",
    )
    manifest["files"] = {
        str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()
    }
    (destination / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(destination))
    print(f"Prepared {sum(len(c['pages']) for c in manifest['cases'])} matched pages: {archive}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    prepare(parser.parse_args().output_directory)
