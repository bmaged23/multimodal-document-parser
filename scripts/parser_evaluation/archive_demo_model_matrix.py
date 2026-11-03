"""Freeze completed shared-Gemma outputs separately from all original OCR runs."""

import json
import subprocess
from pathlib import Path

from multimodal_doc_parser.document_schema.document import Document

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "paddle": "PaddleOCR-VL-1.6",
    "dots": "dots.mocr-Q4_K_M-mmproj-Q8_0",
    "bf16": "dots.mocr-original-HF-BF16-Colab-T4-SDPA",
    "unlimited": "Unlimited-OCR-2-NF4-FP16-vision",
}
RAW = {
    "paddle": "paddleocr",
    "dots": "dots_mocr",
    "bf16": "dots_mocr",
    "unlimited": "unlimited_ocr_2",
}


def main() -> None:
    plan = json.loads((ROOT / "evaluation/demo-model-matrix.json").read_text())
    status = json.loads(
        (ROOT / "outputs/evaluation_batches/four-model-demo-enrichment-01.json").read_text()
    )
    raw_status = json.loads(
        (ROOT / "outputs/parser_runs/unlimited-ocr-2-nf4-four-model-demo-01.json").read_text()
    )
    if status["status"] != "completed" or raw_status["status"] not in {
        "completed",
        "completed_with_errors",
    }:
        raise ValueError("Wait for model and Gemma workers to finish before freezing the matrix")
    registry_path = ROOT / "evaluation/preserved-runs.json"
    registry = json.loads(registry_path.read_text())
    identities = []
    for row in plan["rows"]:
        run = ROOT / "outputs/parser_runs" / row["output_directory"]
        document = Document.model_validate_json((run / "document.json").read_text())
        identity = f"demo_v5_{row['case_key']}_{row['profile']}_gemma_01"
        identities.append(identity)
        if any(c["id"] == identity for c in registry["cases"]):
            continue
        entry = dict(
            id=identity,
            model_name=MODELS[row["profile"]] + "+Gemma4-31b-OllamaCloud",
            run_directory=str(run.relative_to(ROOT)),
            raw_results_subdirectory=RAW[row["profile"]],
            expected_evaluation_pages=len(document.pages),
            evaluation_pages=list(range(1, len(document.pages) + 1)),
            related_run_directories=["outputs/parser_runs/" + row["source_directory"]],
            reproduction_files=[
                "evaluation/demo-model-matrix.json",
                "scripts/visual_evaluation/enrich_demo_model_matrix.py",
            ],
            notes="Shared Gemma enrichment of this OCR profile's own detected visual regions. "
            "Original OCR predictions retained separately. Generated interpretations and "
            "source checks are not independent ground truth. Failure denominators retained.",
        )
        registry["cases"].append(entry)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    for identity in identities:
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/parser_evaluation/archive_evaluation_runs.py"),
                "--case",
                identity,
            ],
            cwd=ROOT,
            check=True,
        )
    print("All sixteen enriched output snapshots preserved")


if __name__ == "__main__":
    main()
