"""Refuse an incomplete four-document/model matrix before packaging the public demo."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.public_demo import build_public_demo
from multimodal_doc_parser.evaluation.showcase import build_showcase

ROOT = Path(__file__).resolve().parents[2]
FINISHED = {"completed", "completed_with_errors", "completed_with_warnings", "not_required"}
PROFILES = {"paddle", "dots", "bf16", "unlimited"}


def validate_matrix() -> dict:
    plan = json.loads((ROOT / "evaluation/demo-model-matrix.json").read_text())
    rows = plan["rows"]
    cases = {row["case_id"] for row in rows}
    if len(cases) != 4 or len(rows) != 16:
        raise ValueError("The public comparison requires exactly four PDFs and four profiles")
    missing, coverage = [], []
    identities: dict[str, str] = {}
    for case in cases:
        if {r["profile"] for r in rows if r["case_id"] == case} != PROFILES:
            raise ValueError("Each PDF must have the same four execution profiles")
    for row in rows:
        run = ROOT / "outputs/parser_runs" / row["output_directory"]
        source = ROOT / "outputs/parser_runs" / row["source_directory"]
        if row["profile"] == "unlimited":
            source = source.with_name(source.name + "-canonical-demo-01")
        if not (run / "enrichment-manifest.json").is_file():
            missing.append(f"{row['case_id']} / {row['profile']}")
            continue
        enrichment = json.loads((run / "enrichment-manifest.json").read_text())
        if enrichment["status"] not in FINISHED or enrichment.get("failed_visuals", 0):
            missing.append(f"{row['case_id']} / {row['profile']} (visuals incomplete)")
            continue
        original = Document.model_validate_json((source / "document.json").read_text())
        document = Document.model_validate_json((run / "document.json").read_text())
        if document.document_id != row["source_document_id"]:
            raise ValueError("Output differs from the frozen source-PDF selection")
        if original.document_id != document.document_id or original.pages != document.pages:
            raise ValueError("Enrichment source identity or page geometry changed")
        source_hash = hashlib.sha256((run / "source.pdf").read_bytes()).hexdigest()
        if document.document_id != "sha256:" + source_hash:
            raise ValueError("Demo PDF bytes differ from their source identity")
        case = row["case_id"]
        if case in identities and identities[case] != document.document_id:
            raise ValueError("Models used different source PDFs")
        identities[case] = document.document_id
        source_elements = {element.id: element for element in original.elements}
        if set(source_elements) != {e.id for e in document.elements}:
            raise ValueError("Gemma changed the OCR element inventory")
        for element in document.elements:
            before = source_elements[element.id]
            if (
                element.source_regions != before.source_regions
                or element.provenance != before.provenance
            ):
                raise ValueError("Gemma changed OCR source coordinates or producer")
            if element.type not in {"chart", "image"} and element != before:
                raise ValueError("Gemma changed OCR text or table content")
        coverage.append(
            dict(
                case=case,
                profile=row["profile"],
                pages=len(document.pages),
                visual_descriptions=enrichment.get("completed_visuals", 0),
                status=enrichment["status"],
            )
        )
    if missing:
        raise ValueError("Unfinished matrix cells; do not publish: " + ", ".join(missing))
    if len(set(identities.values())) != 4:
        raise ValueError("The four cases must contain four distinct source PDFs")
    return dict(
        status="ready",
        cases=4,
        profiles=4,
        cells=coverage,
        scope="Matched saved outputs; completeness is not accuracy or an overall ranking",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--version", default="5.3")
    parser.add_argument("--evaluation-directory", type=Path)
    args = parser.parse_args()
    coverage = validate_matrix()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    full = build_showcase(
        ROOT, output / "showcase", ROOT / "evaluation/four-model-enriched-showcase-cases.json"
    )
    selection = json.loads((ROOT / "evaluation/public-demo-selection.json").read_text())
    selection["version"] = args.version
    path = output / "public-selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n")
    public = build_public_demo(
        full.parent, output / "public-demo", path, evidence_directory=ROOT / "evaluation/results"
    ).parent
    if args.evaluation_directory:
        evaluation = public / "evaluation"
        evaluation.mkdir()
        # Publish metrics only; source transcriptions and expected/actual traces stay private.
        for name in ["metrics.json", "page-metrics.json", "excluded-references.json"]:
            shutil.copy2(args.evaluation_directory / name, evaluation / name)
        spec = importlib.util.spec_from_file_location(
            "manual_reference_report",
            ROOT / "scripts/visualization/build_manual_reference_report.py",
        )
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        report.build(evaluation, demo_directory=public)
        index = public / "index.html"
        html = index.read_text().replace(
            '<p id="model-choice" hidden>',
            '<p><a id="evaluation-report-link" href="evaluation/" '
            'style="font-size:13px;font-weight:650">'
            "70-page evaluation · ranking and metrics ↗</a></p>"
            '<p id="model-choice" hidden>',
            1,
        )
        index.write_text(html)
        manifest_path = public / "build-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["evaluation"] = dict(
            reference_pages=70,
            page_profile_slots=280,
            scope="Assistant source-reference development study; qualitative overall ranking",
            private_reference_transcriptions_included=False,
            private_failure_traces_included=False,
        )
        manifest["artifacts"] = {
            item.relative_to(public).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in public.rglob("*")
            if item.is_file() and item != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output / f"public-demo-v{args.version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for item in sorted(public.rglob("*")):
            if item.is_file():
                zipped.write(item, item.relative_to(public))
    with zipfile.ZipFile(archive) as zipped:
        if (
            len(zipped.infolist()) > 2000
            or sum(m.file_size for m in zipped.infolist()) > 500 * 1024**2
        ):
            raise ValueError("Bundle exceeds the current verified Pages deployment limits")
    print(
        json.dumps(
            dict(archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        )
    )


if __name__ == "__main__":
    main()
