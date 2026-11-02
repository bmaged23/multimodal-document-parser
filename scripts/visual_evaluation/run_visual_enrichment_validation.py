"""Wait for local model setup, enrich the original PDF, and preserve its evaluation."""

import argparse
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

from multimodal_doc_parser.visual_enrichment.ollama import api, enrich_document, save

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemma4:31b"
DESTINATION = ROOT / "outputs/parser_runs/paddleocr-qwen3-vl-visual-enrichment-01"


def main() -> None:
    global DESTINATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="01")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--provider", choices=["ollama", "qwen-cloud", "ollama-cloud"], default="ollama-cloud"
    )
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if not args.suffix.isalnum():
        parser.error("Suffix must contain letters and digits only")
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", args.model).strip("-")
    prefix = f"paddleocr-{model_slug}-visual-enrichment"
    DESTINATION = ROOT / f"outputs/parser_runs/{prefix}-{args.suffix}"
    if DESTINATION.exists():
        raise ValueError("Validation destination exists; preserve it and choose a new run")
    progress = ROOT / f"outputs/parser_runs/{model_slug}-visual-validation-{args.suffix}.json"
    if args.provider == "qwen-cloud":
        from multimodal_doc_parser.visual_enrichment.qwen_cloud import QwenCloudClient

        QwenCloudClient(args.model, args.base_url)
    elif args.provider == "ollama-cloud":
        from multimodal_doc_parser.visual_enrichment.ollama_cloud import OllamaCloudClient

        OllamaCloudClient(args.model)
    else:
        save(progress, {"status": "waiting_for_model", "model": args.model})
        for _ in range(360):
            try:
                tags = api("http://127.0.0.1:11435", "/api/tags", timeout=3)
                if any(m["name"] == args.model for m in tags.get("models", [])):
                    break
            except OSError:
                pass
            time.sleep(5)
        else:
            save(progress, {"status": "setup_failed", "error": "Model not ready within 30 minutes"})
            raise TimeoutError("Model setup incomplete")
    save(progress, {"status": "running", "model": args.model, "run_directory": str(DESTINATION)})
    result = enrich_document(
        ROOT / "outputs/parser_runs/paddleocr-vl-1.6-reviewed",
        DESTINATION,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url or "http://127.0.0.1:11435",
    )
    document = json.loads((result / "document.json").read_text())
    reference = {
        "document_id": document["document_id"],
        "annotation_provenance": "Coding assistant source-image spot checks on the synthetic "
        "two-page original: three printed chart labels and three flowchart connections. "
        "Not independent human annotations or a full accuracy benchmark. "
        "Reference was not supplied to the vision model.",
        "visual_types": [
            {"element_id": "element_00010", "kind": "chart"},
            {"element_id": "element_00024", "kind": "flowchart"},
            {"element_id": "element_00026", "kind": "image"},
        ],
        "chart_points": [
            {"element_id": "element_00010", "label": label, "value": value, "unit": "ms"}
            for label, value in [("35%", 18.4), ("55%", 23.1), ("75%", 31.7)]
        ],
        "flow_edges": [
            {
                "element_id": "element_00024",
                "source": "Read sensor",
                "target": "Error > 3%?",
                "label": "",
            },
            {
                "element_id": "element_00024",
                "source": "Error > 3%?",
                "target": "Flag for review",
                "label": "YES",
            },
            {
                "element_id": "element_00024",
                "source": "Error > 3%?",
                "target": "Accept reading",
                "label": "NO",
            },
        ],
    }
    save(result / "source-derived-visual-reference.json", reference)
    spec = importlib.util.spec_from_file_location(
        "visual_scoring", ROOT / "scripts/visual_evaluation/evaluate_visual_enrichment.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    checks = scorer.evaluate(result, reference)
    save(result / "visual-spot-checks.json", checks)
    manifest = json.loads((result / "enrichment-manifest.json").read_text())
    registry_path = ROOT / "evaluation/preserved-runs.json"
    registry = json.loads(registry_path.read_text())
    case = f"paddleocr_{model_slug.replace('-', '_')}_original_visual_enrichment_{args.suffix}"
    registry["cases"].append(
        {
            "id": case,
            "run_directory": str(result.relative_to(ROOT)),
            "expected_evaluation_pages": 2,
            "evaluation_pages": [1, 2],
            "raw_results_subdirectory": "paddleocr",
            "model_name": f"PaddleOCR-VL-1.6+{args.model}",
            "notes": "Original Paddle extraction plus generated Ollama visual enrichment. "
            "Check enrichment-manifest separately from OCR run-status for visual failures. "
            "Not verified reference annotations.",
        }
    )
    save(registry_path, registry)
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/parser_evaluation/archive_evaluation_runs.py"),
            "--case",
            case,
        ],
        cwd=ROOT,
        check=True,
    )
    save(
        progress,
        {
            "status": manifest["status"],
            "run_directory": str(result),
            "completed_visuals": manifest["completed_visuals"],
            "failed_visuals": manifest["failed_visuals"],
            "selected_checks_passed": checks["passed"],
            "selected_checks_total": checks["checked"],
        },
    )
    print(json.dumps(json.loads(progress.read_text())), flush=True)


if __name__ == "__main__":
    main()
