"""Reuse saved OCR and enrich each model's own visual regions with cloud Gemma."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from multimodal_doc_parser.parser_adapters.unlimited_saved import normalize_saved_run
from multimodal_doc_parser.visual_enrichment.ollama import enrich_document

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/parser_runs"
STATE = ROOT / "outputs/evaluation_batches/four-model-demo-enrichment-01.json"
FINISHED = {"completed", "completed_with_errors", "completed_with_warnings", "not_required"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def step(row: dict) -> dict:
    source = RUNS / row["source_directory"]
    if not source.exists():
        return {**row, "status": "waiting_for_ocr"}
    if row["profile"] == "unlimited":
        status = read(source / "run-status.json")
        if status.get("status") not in FINISHED:
            return {**row, "status": "waiting_for_ocr"}
        normalized = RUNS / (row["source_directory"] + "-canonical-demo-01")
        if not normalized.exists():
            normalize_saved_run(source, normalized)
        source = normalized
    document = read(source / "document.json")
    if document["status"] not in {"parsed", "parsing"}:
        return {**row, "status": "waiting_for_ocr"}
    destination = RUNS / row["output_directory"]
    if destination.exists():
        saved = read(destination / "document.json")
        if saved["document_id"] != document["document_id"]:
            raise ValueError("Saved Gemma output/source identity mismatch")
        state = read(destination / "enrichment-manifest.json")
        if state["status"] in FINISHED:
            return {
                **row,
                "status": state["status"],
                "reused": True,
                "completed_visuals": state.get("completed_visuals", 0),
                "failed_visuals": state.get("failed_visuals", 0),
            }
        raise ValueError("Interrupted enrichment retained; select a fresh output attempt")
    visuals = [e for e in document["elements"] if e["type"] in {"image", "chart"}]
    if not visuals:
        shutil.copytree(source, destination)
        state = dict(
            status="not_required",
            total_visuals=0,
            completed_visuals=0,
            failed_visuals=0,
            source_run=str(source),
            notes="No OCR-detected visual regions; no Gemma request.",
        )
        write(destination / "enrichment-manifest.json", state)
    else:
        enrich_document(
            source,
            destination,
            model="gemma4:31b",
            provider="ollama-cloud",
            allow_partial=True,
            response_cache=ROOT / ".cache/gemma_visual_requests",
        )
        state = read(destination / "enrichment-manifest.json")
    os.sync()
    return {
        **row,
        "status": state["status"],
        "reused": False,
        "completed_visuals": state.get("completed_visuals", 0),
        "failed_visuals": state.get("failed_visuals", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    plan = read(ROOT / "evaluation/demo-model-matrix.json")
    state = dict(status="running", rows=[])
    while True:
        for row in plan["rows"]:
            state["current"] = dict(case=row["case_id"], profile=row["profile"])
            write(STATE, state)
            print(f"Checking {row['case_id']} / {row['profile']}", flush=True)
            result = step(row)
            state["rows"] = [
                r
                for r in state["rows"]
                if (r["case_id"], r["profile"]) != (row["case_id"], row["profile"])
            ] + [result]
            write(STATE, state)
        waiting = [r for r in state["rows"] if r["status"] == "waiting_for_ocr"]
        state.update(status="waiting_for_ocr" if waiting else "completed", current=None)
        write(STATE, state)
        if not args.watch or not waiting:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
