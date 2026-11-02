"""Billing failures are availability failures, not bad model-quality scores."""

import importlib.util
import json
import urllib.error
from pathlib import Path

from multimodal_doc_parser.document_schema.document import VisualAnalysis


def module():
    spec = importlib.util.spec_from_file_location(
        "vlm_benchmark", "scripts/visual_evaluation/run_vlm_benchmark.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_payment_failure_stops_requests_and_never_invents_quality_scores(tmp_path, monkeypatch):
    benchmark = module()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(dict(visuals=[dict(page_number=1, kind="chart")])))
    config = dict(
        benchmark_id="fixture",
        source_document_id="sha256:" + "0" * 64,
        source_cached_run=str(tmp_path / "unused"),
        models=["paid-model"],
        reuse_cached_model="different-model",
        reference=str(reference),
        protocol="Synthetic test",
        cases=[
            dict(id=f"case-{i}", element_id=f"e{i}", kind="chart", pdf_page=1) for i in range(5)
        ],
    )
    calls = []

    class Client:
        base_url = "https://ollama.com"

        def __init__(self, *args):
            pass

        def api(self, *args, **kwargs):
            calls.append(args)
            raise urllib.error.HTTPError(
                "https://ollama.com/api/chat", 402, "Payment Required", {}, None
            )

    monkeypatch.setattr(benchmark, "OllamaCloudClient", Client)
    monkeypatch.setattr(benchmark, "render_report", lambda *args: None)
    monkeypatch.setattr(
        benchmark, "request_from_cache", lambda *args: dict(prompt="fixture", images=[])
    )
    destination = tmp_path / "results"
    destination.mkdir()
    result = benchmark.run(config, destination)
    assert len(calls) == 1
    assert result["status"] == "blocked_by_account_access" and result["ranking"] is None
    assert result["models"][0]["status"] == "unavailable"
    assert all(
        c["status"] == "unavailable" and not c["stages"] for c in result["models"][0]["cases"]
    )


def test_series_assignments_and_numeric_abstention_have_distinct_scores():
    benchmark = module()
    analysis = VisualAnalysis(
        kind="chart",
        description="Fixture",
        series=[
            dict(
                name="A",
                unit="gigawatt",
                points=[dict(label="2024", value=10, value_basis="printed_label")],
            )
        ],
    )
    expected = dict(
        kind="chart",
        printed_points=[dict(series="A", label="2024", value=20)],
        series_units={"A": "GW"},
        no_numeric_values=True,
    )
    score = benchmark.score(analysis, expected)
    assert score["printed_points"] == dict(passed=0, checked=1)
    assert score["series_units"] == dict(passed=1, checked=1)
    assert score["numeric_abstention"] is False


def test_five_case_selection_contains_three_different_charts_and_two_photos():
    config = json.loads(Path("evaluation/vlm_benchmarks/irena-five-visuals-v1.json").read_text())
    assert len(config["models"]) == 5 and len(set(config["models"])) == 5
    assert len(config["cases"]) == 5
    assert sum(c["kind"] == "chart" for c in config["cases"]) == 3
    assert sum(c["kind"] == "image" for c in config["cases"]) == 2
    assert {"multi-series-line-chart", "grouped-bar-chart", "unlabelled-scatter-plot"} <= {
        c["id"] for c in config["cases"]
    }
