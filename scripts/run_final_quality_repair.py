"""Replay saved OCR and run one bounded visual repair batch; never launch GPU OCR."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from archive_evaluation_runs import snapshot
from evaluate_visual_enrichment import evaluate as evaluate_visual_spots

from multimodal_doc_parser.evaluation.source_review import evaluate_source_review
from multimodal_doc_parser.parser_adapters.parsing_run import parse_prepared_document
from multimodal_doc_parser.visual_enrichment.ollama import enrich_document

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def register(case: dict) -> None:
    path = ROOT / "evaluation/preserved-runs.json"
    registry = json.loads(path.read_text())
    if any(c["id"] == case["id"] for c in registry["cases"]):
        raise ValueError("Preservation case already exists")
    registry["cases"].append(case)
    save(path, registry)
    snapshot(case, ROOT / registry["archive_root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    args.output_directory.mkdir(parents=True, exist_ok=False)
    save(args.output_directory / "configuration.json", configuration)
    state = dict(
        status="running",
        started_unix=time.time(),
        mode="saved_raw_replay_no_gpu_ocr",
        note="One draft and one review call per chart. No automatic inference retries. "
        "References are read only AFTER enrichment, never sent to a model.",
        cases={},
    )
    for case in configuration["cases"]:
        entry = state["cases"][case["id"]] = dict(stage="raw_replay")
        baseline = ROOT / case["baseline_run"]
        source_hashes = {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                baseline / "document.json",
                *sorted((baseline / "raw/paddleocr").glob("*.json")),
            ]
        }
        entry["baseline_hashes"] = source_hashes
        save(args.output_directory / "status.json", state)
        try:
            run = ROOT / case["replayed_run"]
            parse_prepared_document(
                ROOT / case["prepared_bundle"],
                run,
                ROOT / "backends/paddleocr_runtime",
                raw_results_directory=baseline / "raw/paddleocr",
            )
            entry["stage"] = "visual_enrichment"
            save(args.output_directory / "status.json", state)
            enriched = enrich_document(
                run, ROOT / case["enriched_run"], provider="ollama-cloud", model="gemma4:31b"
            )
            entry["stage"] = "source_evaluation"
            save(args.output_directory / "status.json", state)
            # No reference file has been opened before this point.
            reference = ROOT / case["reference"]
            if case.get("reference_kind") == "visual_spots":
                result = evaluate_visual_spots(enriched, json.loads(reference.read_text()))
            else:
                result = evaluate_source_review(enriched, reference)
            result_path = args.output_directory / f"{case['id']}-results.json"
            save(result_path, result)
            for mode, directory, model in [
                ("ocr", run, "PaddleOCR-VL-1.6"),
                ("enriched", enriched, "PaddleOCR-VL-1.6+gemma4:31b"),
            ]:
                record = dict(
                    id=f"{case['id']}-final-repair-{mode}-02",
                    model_name=model,
                    run_directory=str(directory.relative_to(ROOT)),
                    expected_evaluation_pages=case["pages"],
                    evaluation_pages=list(range(1, case["pages"] + 1)),
                    notes="Final repair: replayed identical saved raw OCR; no fresh GPU "
                    "inference. Inferred header roles and source image recovery are "
                    "postprocessing, not raw OCR accuracy. Original baselines retained.",
                )
                if mode == "enriched":
                    record.update(
                        reference_annotations=str(reference.relative_to(ROOT)),
                        comparison_report=str(result_path.relative_to(ROOT)),
                    )
                register(record)
            manifest = json.loads((enriched / "enrichment-manifest.json").read_text())
            entry.update(
                stage="completed",
                enrichment_status=manifest["status"],
                completed_visuals=manifest["completed_visuals"],
                failed_visuals=manifest["failed_visuals"],
                visuals_needing_review=manifest.get("visuals_needing_review", []),
            )
        except Exception as exc:
            entry.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            unchanged = all(digest(ROOT / p) == sha for p, sha in source_hashes.items())
            entry["baseline_unchanged"] = unchanged
            if not unchanged:
                entry.update(stage="failed", error="Baseline unexpectedly changed")
            save(args.output_directory / "status.json", state)
    state["status"] = (
        "completed_with_issues"
        if any(
            c["stage"] == "failed" or c.get("failed_visuals") or c.get("visuals_needing_review")
            for c in state["cases"].values()
        )
        else "completed"
    )
    state["finished_unix"] = time.time()
    save(args.output_directory / "status.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
