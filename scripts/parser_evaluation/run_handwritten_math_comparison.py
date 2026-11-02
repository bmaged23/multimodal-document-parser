"""Prepare four author-provided photos and run the retained GPU parser profiles.

No model-generated reference, source correction, or cloud VLM is used here.
Original JPEGs remain local; public export is a separate, explicit build step.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.parser_adapters.parsing_run import parse_prepared_document
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document

ROOT = Path(__file__).resolve().parents[2]
CASE = "author-handwritten-mathematics-four-pages-01"
SELECTION = ROOT / "evaluation/test_cases/author-handwritten-mathematics-four-pages.json"
NAMES = [
    "WhatsApp Image 2026-11-02 at 1.44.41 AM (1).jpeg",
    "WhatsApp Image 2026-11-02 at 1.44.41 AM.jpeg",
    "WhatsApp Image 2026-11-02 at 1.44.41 AM (3).jpeg",
    "WhatsApp Image 2026-11-02 at 1.44.41 AM (2).jpeg",
]
RUNS = {
    "paddle": "paddleocr-author-handwritten-mathematics-four-pages-01",
    "dots": "dots-mocr-q4-k-m-author-handwritten-mathematics-four-pages-01",
    "unlimited": "unlimited-ocr-2-nf4-author-handwritten-mathematics-four-pages-01",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_script(name: str):
    path = ROOT / "scripts/parser_evaluation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(downloads: Path) -> Path:
    source = ROOT / "data/input_documents/evaluation_corpus/author_handwritten_mathematics"
    source.mkdir(parents=True, exist_ok=True)
    originals = []
    for number, name in enumerate(NAMES, 1):
        supplied = downloads / name
        retained = source / f"assignment_page_{number:02d}.jpeg"
        if retained.exists():
            if digest(retained) != digest(supplied):
                raise ValueError("Original image changed; use a new case instead of overwriting")
        else:
            shutil.copyfile(supplied, retained)
        with Image.open(retained) as image:
            width, height = image.size
        originals.append(
            {
                "page": number,
                "filename": retained.name,
                "sha256": digest(retained),
                "width": width,
                "height": height,
            }
        )
    pdf = source / "author_handwritten_mathematics_four_pages.pdf"
    if not pdf.exists():
        # JPEG pass-through: no binarization, enhancement, crop, or recompression.
        canvas = Canvas(str(pdf), invariant=1, pageCompression=1)
        canvas.setTitle("Author handwritten mathematics assignment · four pages")
        for original in originals:
            width, height = original["width"] / 2, original["height"] / 2
            canvas.setPageSize((width, height))
            canvas.drawImage(str(source / original["filename"]), 0, 0, width, height)
            canvas.showPage()
        canvas.save()
    bundle = ROOT / "outputs/parsed_documents" / f"{digest(pdf)}-dpi-144"
    if not bundle.exists():
        bundle = prepare_document(pdf, ROOT / "outputs/parsed_documents", dpi=144)
    document = read(bundle / "document.json")
    selection = {
        "case_id": CASE,
        "source_filename": pdf.name,
        "source_sha256": digest(pdf),
        "render_dpi": 144,
        "original_pdf_pages": [1, 2, 3, 4],
        "ordering": "Visible handwritten page numbers 1, 2, 3, 4; not download order",
        "original_images": originals,
        "rendered_images": [
            {"page": page["page_number"], "sha256": digest(bundle / page["image"]["path"])}
            for page in document["pages"]
        ],
        "preprocessing": "Original JPEGs embedded without enhancement; shared PDF rendered once",
        "coverage": ["prose", "set notation", "Venn diagrams", "functions", "truth table"],
        "rights_note": "Author-provided handwritten assignment; author requested public demo use",
        "reference_policy": (
            "Transcribe as written; exclude ambiguous/crossed-out spans from scored references"
        ),
        "prepared_bundle": str(bundle.relative_to(ROOT)),
    }
    if SELECTION.exists() and read(SELECTION) != selection:
        raise ValueError("Selection changed; preserve previous version and use a new case")
    save(SELECTION, selection)
    return bundle


def paddle(bundle: Path) -> None:
    run = ROOT / "outputs/parser_runs" / RUNS["paddle"]
    parse_prepared_document(bundle, run, ROOT / "backends/paddleocr_runtime")
    registry_path = ROOT / "evaluation/preserved-runs.json"
    registry = read(registry_path)
    entry = {
        "id": "paddleocr_author_handwritten_mathematics_four_pages_01",
        "model_name": "PaddleOCR-VL-1.6",
        "run_directory": str(run.relative_to(ROOT)),
        "expected_evaluation_pages": 4,
        "evaluation_pages": [1, 2, 3, 4],
        "selection_manifest": str(SELECTION.relative_to(ROOT)),
        "notes": "New GPU inference on all four author-provided pages; no source correction",
    }
    registry["cases"].append(entry)
    save(registry_path, registry)
    load_script("archive_evaluation_runs").snapshot(entry, ROOT / registry["archive_root"])


def dots(bundle: Path) -> None:
    batch = load_script("run_quantized_evaluation_batch").Batch("author-math-01")
    batch.run(
        cases=[
            {
                "id": "dots_mocr_q4_author_handwritten_mathematics_four_pages_01",
                "name": "Author handwritten mathematics · four pages",
                "prepared_bundle": str(bundle),
                "total_pages": 4,
                "original_pdf_pages": [1, 2, 3, 4],
                "run_directory": str(ROOT / "outputs/parser_runs" / RUNS["dots"]),
                "selection_manifest": str(SELECTION.relative_to(ROOT)),
                "paddle_baseline": str(ROOT / "outputs/parser_runs" / RUNS["paddle"]),
                "comparison_directory": str(
                    ROOT / "outputs/parser_comparisons/author-math-paddle-vs-dots-01"
                ),
            }
        ]
    )


def unlimited(bundle: Path) -> None:
    load_script("run_unlimited_ocr_2_evaluation_batch").main(
        cases=[
            {
                "id": "unlimited_ocr_2_nf4_author_handwritten_mathematics_four_pages_01",
                "name": "Author handwritten mathematics · four pages",
                "bundle": bundle.name,
                "run": RUNS["unlimited"],
                "original_pdf_pages": [1, 2, 3, 4],
            }
        ],
        state_path=ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-author-math-batch-01.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "paddle", "dots", "unlimited", "all"], default="all"
    )
    args = parser.parse_args()
    bundle = prepare(args.downloads)
    print(f"Prepared shared four-page input: {bundle}", flush=True)
    for name, run_stage in [("paddle", paddle), ("dots", dots), ("unlimited", unlimited)]:
        if args.stage in (name, "all"):
            print(f"Starting {name} GPU stage", flush=True)
            run_stage(bundle)
            print(f"Finished {name} GPU stage", flush=True)


if __name__ == "__main__":
    main()
