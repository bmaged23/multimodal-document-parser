"""Compare raw VLM responses on fixed saved crops; never rerun OCR or repair values."""

import argparse
import base64
import hashlib
import json
import shutil
import time
import urllib.error
from html import escape
from pathlib import Path

from multimodal_doc_parser.document_schema.document import VisualAnalysis
from multimodal_doc_parser.evaluation.source_review import normalized
from multimodal_doc_parser.visual_enrichment.ollama import safe_asset
from multimodal_doc_parser.visual_enrichment.ollama_cloud import OllamaCloudClient


def save(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def unit_name(unit: str | None) -> str:
    value = normalized(unit or "")
    return {
        "gigawatt": "gw",
        "gigawatts": "gw",
        "basis points": "bps",
        "us dollars/kwh": "usd/kwh",
        "2024 usd/kwh": "usd/kwh",
    }.get(value, value)


def score(analysis: VisualAnalysis, expected: dict) -> dict:
    points = []
    for wanted in expected.get("printed_points", []):
        actual = [
            p
            for series in analysis.series
            if normalized(series.name) == normalized(wanted["series"])
            for p in series.points
            if normalized(p.label) == normalized(wanted["label"])
        ]
        passed = (
            len(actual) == 1
            and actual[0].value is not None
            and (
                abs(actual[0].value - wanted["value"]) < 1e-9
                and actual[0].value_basis == "printed_label"
            )
        )
        points.append(dict(expected=wanted, actual=[p.model_dump() for p in actual], passed=passed))
    units = []
    for name, unit in expected.get("series_units", {}).items():
        actual = [s.unit for s in analysis.series if normalized(s.name) == normalized(name)]
        units.append(
            dict(
                series=name,
                expected=unit,
                actual=actual,
                passed=len(actual) == 1 and unit_name(actual[0]) == unit_name(unit),
            )
        )
    result = dict(
        kind_correct=analysis.kind == expected["kind"],
        printed_points=dict(passed=sum(p["passed"] for p in points), checked=len(points)),
        series_units=dict(passed=sum(u["passed"] for u in units), checked=len(units)),
        point_checks=points,
        unit_checks=units,
        description=analysis.description,
        uncertainties=analysis.uncertainties,
    )
    if expected.get("no_numeric_values"):
        result["numeric_abstention"] = not any(
            p.value is not None for s in analysis.series for p in s.points
        )
    if expected.get("no_data_points"):
        result["no_invented_categories"] = not any(s.points for s in analysis.series)
    if analysis.kind == "image":
        result["factuality_review"] = "requires source-grounded manual review"
    return result


def parse_response(response: dict) -> VisualAnalysis:
    if not response.get("done") or response.get("done_reason") != "stop":
        raise ValueError("Incomplete model generation")
    return VisualAnalysis.model_validate_json(response["response"])


def request_from_cache(cached: Path, identity: str, model: str) -> dict:
    raw = cached / "raw/visual_enrichment"
    settings = json.loads((raw / f"{identity}-request-settings.json").read_text())
    inputs = json.loads((raw / f"{identity}-input-manifest.json").read_text())["inputs"]
    images = []
    for item in inputs:
        data = safe_asset(raw, item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Cached image input checksum mismatch")
        images.append(base64.b64encode(data).decode())
    return dict(
        model=model,
        prompt=settings["analysis_prompt"],
        format=settings["format"],
        options=settings["options"],
        images=images,
        stream=False,
        think=False,
    )


def run(configuration: dict, destination: Path) -> dict:
    cached = Path(configuration["source_cached_run"])
    records = []
    for model in configuration["models"]:
        if Path(model).name != model or "\\" in model:
            raise ValueError("Cloud model name must be a single directory component")
        folder = destination / "models" / model
        folder.mkdir(parents=True)
        record = dict(
            model=model,
            status="running",
            cases=[],
            cached=model == configuration["reuse_cached_model"],
        )
        records.append(record)
        client = None if record["cached"] else OllamaCloudClient(model)
        account_error = None
        for case in configuration["cases"]:
            item = dict(
                case_id=case["id"], element_id=case["element_id"], kind=case["kind"], stages={}
            )
            record["cases"].append(item)
            if account_error:
                item.update(
                    status="unavailable",
                    http_status=account_error,
                    error="Account access failure; no inference attempted for this case",
                )
                continue
            try:
                request = request_from_cache(cached, case["element_id"], model)
                raw = cached / "raw/visual_enrichment"
                for stage in ["draft", "review"] if case["kind"] == "chart" else ["draft"]:
                    started = time.monotonic()
                    if stage == "review":
                        template = json.loads(
                            (raw / f"{case['element_id']}-review-request-settings.json").read_text()
                        )
                        prefix, remainder = template["prompt"].split(
                            "DRAFT (untrusted model output):\n", 1
                        )
                        _, suffix = remainder.split("\nNATIVE SOURCE TEXT", 1)
                        draft = parse_response(item["stages"]["draft"]["response"])
                        request["prompt"] = (
                            prefix
                            + "DRAFT (untrusted model output):\n"
                            + draft.model_dump_json()
                            + "\nNATIVE SOURCE TEXT"
                            + suffix
                        )
                    safe_settings = {k: v for k, v in request.items() if k != "images"}
                    safe_settings["image_sha256"] = [
                        hashlib.sha256(base64.b64decode(s)).hexdigest() for s in request["images"]
                    ]
                    save(folder / f"{case['element_id']}-{stage}-request.json", safe_settings)
                    if record["cached"]:
                        suffix = "review-response.json" if stage == "review" else "response.json"
                        filename = f"{case['element_id']}-{suffix}"
                        response = json.loads((raw / filename).read_text())
                    else:
                        response = client.api(
                            client.base_url, "/api/generate", request, timeout=120
                        )
                    item["stages"][stage] = dict(
                        response=response,
                        wall_seconds=None if record["cached"] else time.monotonic() - started,
                    )
                    save(folder / f"{case['element_id']}-{stage}-response.json", response)
                    analysis = parse_response(response)
                    item["stages"][stage]["analysis"] = analysis.model_dump()
                item["status"] = "completed"
            except urllib.error.HTTPError as exc:
                item.update(
                    status="unavailable" if exc.code in [401, 402, 403] else "failed",
                    http_status=exc.code,
                    error=exc.reason,
                )
                if exc.code in [401, 402, 403]:
                    account_error = exc.code
            except (OSError, ValueError, KeyError) as exc:
                item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save(folder / "model-status.json", record)
            save(destination / "status.json", dict(status="running", models=records))
        record["status"] = (
            "unavailable"
            if account_error
            else "completed"
            if all(c["status"] == "completed" for c in record["cases"])
            else "completed_with_errors"
        )
        save(folder / "model-status.json", record)
    # Expected answers are opened only after every inference request is finished.
    reference = json.loads(Path(configuration["reference"]).read_text())
    expected_by_page = {v["page_number"]: v for v in reference["visuals"]}
    for record in records:
        for case, item in zip(configuration["cases"], record["cases"], strict=True):
            expected = expected_by_page[case["pdf_page"]]
            for stage in item["stages"].values():
                if "analysis" in stage:
                    stage["quality"] = score(
                        VisualAnalysis.model_validate(stage["analysis"]), expected
                    )
            for stage in item["stages"].values():
                stage.pop("response", None)
    result = dict(
        benchmark_id=configuration["benchmark_id"],
        models=records,
        source_document_id=configuration["source_document_id"],
        reference_sha256=hashlib.sha256(Path(configuration["reference"]).read_bytes()).hexdigest(),
        protocol=configuration["protocol"],
        status="blocked_by_account_access"
        if any(r["status"] == "unavailable" for r in records)
        else "completed",
        ranking=None,
        ranking_note="Unavailable models have null quality, not zero accuracy. "
        "No winner is declared until comparable results and photo reviews exist.",
    )
    save(destination / "results.json", result)
    save(
        destination / "status.json",
        dict(
            status=result["status"],
            models=[dict(model=r["model"], status=r["status"]) for r in records],
        ),
    )
    render_report(result, configuration, destination)
    return result


def render_report(result: dict, configuration: dict, destination: Path) -> None:
    records = result["models"]
    previews = ""
    raw = Path(configuration["source_cached_run"]) / "raw/visual_enrichment"
    assets = destination / "assets"
    assets.mkdir(exist_ok=True)
    for case in configuration["cases"]:
        inputs = json.loads((raw / f"{case['element_id']}-input-manifest.json").read_text())[
            "inputs"
        ]
        crop = next(item for item in inputs if item["path"].endswith("-crop.png"))
        shutil.copy2(safe_asset(raw, crop["path"]), safe_asset(assets, crop["path"]))
        previews += (
            f'<figure><img src="assets/{escape(crop["path"], quote=True)}" '
            f'alt="{escape(case["id"], quote=True)}" style="max-width:100%;max-height:360px">'
            f"<figcaption>{escape(case['id'])} — source PDF page "
            f"{case['original_pdf_page']}</figcaption></figure>"
        )
    rows = ""
    for record in records:
        for item in record["cases"]:
            qualities = [
                f"{name}: {s['quality']['printed_points']['passed']}/"
                f"{s['quality']['printed_points']['checked']} chart values"
                for name, s in item["stages"].items()
                if "quality" in s and item["kind"] == "chart"
            ]
            description = next(
                (s["quality"]["description"] for s in item["stages"].values() if "quality" in s), ""
            )
            message = "; ".join(qualities) or description or item.get("error", "")
            rows += (
                f"<tr><td>{escape(record['model'])}</td><td>{escape(item['case_id'])}</td>"
                f"<td>{escape(item['status'])}</td><td>{escape(message)}</td></tr>"
            )
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five-model VLM comparison</title><style>
body{font:16px system-ui;margin:24px;color:#172538}table{border-collapse:collapse;width:100%}
td,th{padding:10px;border:1px solid #ccd;text-align:left}
aside{padding:15px;background:#fff4d8}</style>
<h1>Five models × five source visuals</h1><aside>Raw model outputs only.
Native-PDF numerical corrections are excluded. Account access failures are not quality failures.
No winner can be declared while models are unavailable.</aside>
<p>Three different chart types and two photographic scenes;
one draft plus one review per chart. Photo factuality requires manual source review.</p>
<p><a href="results.json">Full results and provenance</a></p>
<table><tr><th>Model</th><th>Case</th><th>Status</th><th>Result or description</th></tr>
"""
    (destination / "index.html").write_text(
        html + rows + "</table><h2>Selected source crops</h2>" + previews + "</html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    configuration = json.loads(args.configuration.read_text())
    save(args.output_directory / "configuration.json", configuration)
    result = run(configuration, args.output_directory)
    print(
        json.dumps(
            dict(
                status=result["status"],
                models=[dict(model=r["model"], status=r["status"]) for r in result["models"]],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
