"""Enrichment must preserve extraction, retain failed responses, and release the model."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_exports.readable_exports import document_markdown
from multimodal_doc_parser.document_schema.document import Document, VisualAnalysis
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document
from multimodal_doc_parser.visual_enrichment import ollama


@pytest.mark.parametrize("provider", ["ollama", "qwen-cloud", "ollama-cloud"])
def test_enrichment_keeps_source_and_partial_failure_traceable(
    tmp_path: Path, monkeypatch, provider
):
    pdf = tmp_path / "source.pdf"
    canvas = Canvas(str(pdf), pagesize=(200, 300))
    canvas.drawString(20, 260, "fixture chart")
    canvas.save()
    source = prepare_document(pdf, tmp_path / "prepared", dpi=72)
    data = json.loads((source / "document.json").read_text())
    data["status"] = "parsed"
    data["elements"].append(
        {
            "id": "heading",
            "type": "heading",
            "level": 2,
            "text": "6. Flowchart",
            "source_regions": [
                {"page_number": 1, "bounding_box": {"x0": 0.1, "y0": 0.01, "x1": 0.8, "y1": 0.08}}
            ],
            "provenance": {"producer": "fixture-ocr", "method": "extracted"},
        }
    )
    for i in [1, 2]:
        Image.new("RGB", (80, 50), "white").save(source / f"assets/crop{i}.png")
        data["elements"].append(
            {
                "id": f"visual_{i}",
                "type": "image",
                "asset": {"path": f"assets/crop{i}.png"},
                "source_regions": [
                    {"page_number": 1, "bounding_box": {"x0": 0.1, "y0": 0.1, "x1": 0.8, "y1": 0.8}}
                ],
                "provenance": {"producer": "fixture-ocr", "method": "extracted"},
            }
        )
    (source / "document.json").write_text(json.dumps(data))
    source_bytes = (source / "document.json").read_bytes()
    requests = []
    analysis = VisualAnalysis(
        kind="chart",
        description="A bar labelled 18.4 ms.",
        series=[
            {
                "name": "latency",
                "unit": "ms",
                "points": [{"label": "35%", "value": 18.4, "value_basis": "printed_label"}],
            }
        ],
    )

    def fake_api(base, route, payload=None, timeout=300):
        if route == "/api/tags":
            return {"models": [{"name": "fixture", "digest": "sha256:fixture"}]}
        if route == "/api/version":
            return {"version": "fixture"}
        if route == "/api/ps":
            return {"models": [{"name": "fixture", "size_vram": 1024}]}
        requests.append(payload)
        if payload.get("keep_alive") == 0:
            return {"done": True}
        if payload["format"].get("required") == ["kind"]:
            return {"done": True, "done_reason": "stop", "response": '{"kind":"chart"}'}
        if len(requests) == 2:
            return {"done": True, "done_reason": "stop", "response": analysis.model_dump_json()}
        return {"done": True, "done_reason": "length", "response": '{"description":'}

    monkeypatch.setattr(ollama, "api", fake_api)
    monkeypatch.setattr(
        "multimodal_doc_parser.visual_enrichment.qwen_cloud.QwenCloudClient",
        lambda *a: SimpleNamespace(base_url="https://cloud.fixture", api=fake_api),
    )
    monkeypatch.setattr(
        "multimodal_doc_parser.visual_enrichment.ollama_cloud.OllamaCloudClient",
        lambda *a: SimpleNamespace(
            base_url="https://ollama.com",
            api=fake_api,
            request=lambda route: {
                "models": [
                    {"name": "fixture", "digest": "sha256:fixture", "capabilities": ["vision"]}
                ]
            },
        ),
    )
    result = ollama.enrich_document(
        source, tmp_path / "enriched", model="fixture", provider=provider
    )
    assert (source / "document.json").read_bytes() == source_bytes
    document = Document.model_validate_json((result / "document.json").read_text())
    visual = next(e for e in document.elements if e.id == "visual_1")
    assert visual.provenance.method == "extracted"
    assert visual.description.provenance.method == "generated"
    assert visual.description.analysis.series[0].points[0].value == 18.4
    assert next(e for e in document.elements if e.id == "visual_2").description is None
    state = json.loads((result / "enrichment-manifest.json").read_text())
    assert state["status"] == "completed_with_errors"
    assert state["completed_visuals"] == 1 and state["failed_visuals"] == 1
    assert (result / "raw/visual_enrichment/visual_2-response.json").exists()
    if provider == "ollama":
        assert requests[-1]["keep_alive"] == 0
    else:
        assert all(request.get("keep_alive") != 0 for request in requests)
        assert "seed" not in state["settings"] and "num_ctx" not in state["settings"]
    first = state["visuals"][0]["inputs"][0]
    content = (result / "raw/visual_enrichment" / first["path"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == first["sha256"]
    assert base64.b64decode(requests[0]["images"][0]) == content
    assert "Model-reported chart values" in document_markdown(document)
    expected_identity = (
        "Qwen Cloud fixture" if provider == "qwen-cloud" else "fixture@sha256:fixture"
    )
    assert expected_identity in (result / "review.html").read_text()


def test_invalid_flow_edges_and_unknown_numeric_values_are_rejected():
    with pytest.raises(ValueError, match="missing node"):
        VisualAnalysis(
            kind="flowchart",
            description="A flow",
            nodes=[{"id": "a", "label": "A"}],
            edges=[{"source": "a", "target": "missing"}],
        )
    with pytest.raises(ValueError, match="Only chart"):
        VisualAnalysis(
            kind="image", description="A photo", series=[{"name": "wrong", "points": []}]
        )


def test_chart_review_retains_draft_and_final_response_without_reference_inputs(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "source.pdf"
    canvas = Canvas(str(pdf), pagesize=(200, 300))
    canvas.drawString(30, 200, "0.100")
    canvas.save()
    source = prepare_document(pdf, tmp_path / "prepared", dpi=72)
    data = json.loads((source / "document.json").read_text())
    data["status"] = "parsed"
    Image.new("RGB", (100, 100), "white").save(source / "assets/chart.png")
    data["elements"] = [
        dict(
            id="chart",
            type="chart",
            asset=dict(path="assets/chart.png"),
            source_regions=[dict(page_number=1, bounding_box=dict(x0=0.1, y0=0.1, x1=0.9, y1=0.9))],
            provenance=dict(producer="fixture", method="extracted"),
        )
    ]
    (source / "document.json").write_text(json.dumps(data))
    before = (source / "document.json").read_bytes()
    prompts = []

    def fake_api(base, route, payload=None, timeout=300):
        if route == "/api/tags":
            return dict(models=[dict(name="fixture", digest="fixture")])
        if route == "/api/version":
            return dict(version="fixture")
        if route == "/api/ps":
            return dict(models=[])
        if payload.get("keep_alive") == 0:
            return dict(done=True)
        prompts.append(payload["prompt"])
        value = 0.1 if "REVIEW PASS" in payload["prompt"] else 0.2
        analysis = VisualAnalysis(
            kind="chart",
            plot_type="bar",
            description="A test chart",
            series=[
                dict(
                    name="A",
                    unit="ms",
                    points=[dict(label="2024", value=value, value_basis="printed_label")],
                )
            ],
        )
        return dict(done=True, done_reason="stop", response=analysis.model_dump_json())

    monkeypatch.setattr(ollama, "api", fake_api)
    result = ollama.enrich_document(source, tmp_path / "result", model="fixture", provider="ollama")
    raw = result / "raw/visual_enrichment"
    draft = json.loads(json.loads((raw / "chart-response.json").read_text())["response"])
    final = json.loads(json.loads((raw / "chart-review-response.json").read_text())["response"])
    assert draft["series"][0]["points"][0]["value"] == 0.2
    assert final["series"][0]["points"][0]["value"] == 0.1
    document = Document.model_validate_json((result / "document.json").read_text())
    assert document.elements[0].description.analysis.series[0].points[0].value == 0.1
    assert document.elements[0].quality_review.status == "model_reviewed"
    assert len(prompts) == 2 and "NATIVE SOURCE TEXT" in prompts[-1]
    assert (source / "document.json").read_bytes() == before
