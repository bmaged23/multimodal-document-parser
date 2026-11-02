"""Real API/worker boundaries without network, OCR or model downloads."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.pipeline_services.service_api import create_service
from multimodal_doc_parser.pipeline_services.service_runtime import work_once
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.storage import JobStore


def test_upload_prepare_job_idempotency_and_rag_api_separation(tmp_path):
    settings = ServiceSettings(
        storage=tmp_path / "storage", api_key="synthetic-service-key-not-a-real-secret"
    )
    client = TestClient(create_service("documents", settings))
    assert client.get("/health/ready").status_code == 200
    assert client.post("/v1/jobs", json={}).status_code == 401
    client.headers["Authorization"] = "Bearer " + settings.api_key
    pdf = tmp_path / "sample.pdf"
    canvas = Canvas(str(pdf))
    canvas.drawString(10, 100, "Synthetic API test")
    canvas.save()
    response = client.post(
        "/v1/documents", files={"file": ("sample.pdf", pdf.read_bytes(), "application/pdf")}
    )
    assert response.status_code == 201
    uploaded = response.json()["artifact_id"]
    body = dict(source_artifact_id=uploaded, dpi=72)
    first = client.post("/v1/jobs", json=body, headers={"Idempotency-Key": "same-upload"})
    assert first.status_code == 202
    assert (
        client.post("/v1/jobs", json=body, headers={"Idempotency-Key": "same-upload"}).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/jobs", json={**body, "dpi": 100}, headers={"Idempotency-Key": "same-upload"}
        ).status_code
        == 409
    )
    assert work_once("documents", settings)
    status = client.get(first.json()["status_url"]).json()
    assert status["status"] == "completed"
    artifact = status["result"]["artifact_id"]
    assert client.get(f"/v1/artifacts/{artifact}/files/document.json").status_code == 200
    assert client.post("/v1/jobs", json=dict(source_artifact_id="../../.env")).status_code == 422
    rag = create_service("rag", settings)
    paths = rag.openapi()["paths"]
    assert "/v1/rag/runs" in paths
    assert "/v1/documents" not in paths and "/v1/jobs" not in paths
    assert "HTTPBearer" in rag.openapi()["components"]["securitySchemes"]


def test_expired_worker_not_retried_and_claim_is_exclusive(tmp_path):
    store = JobStore(tmp_path, "chunking")
    job, _ = store.enqueue(dict(source_artifact_id="a" * 32), "fixture")
    assert store.claim("worker-one")["id"] == job["id"]
    assert store.claim("worker-two") is None
    with store.connect() as connection:
        connection.execute("UPDATE jobs SET heartbeat=? WHERE id=?", (time.time() - 200, job["id"]))
    assert store.claim("worker-two") is None
    assert store.get(job["id"])["status"] == "interrupted"
    assert "No automatic" in store.get(job["id"])["error"]


def test_public_pipeline_has_typed_stage_routes_and_preserves_status_links(tmp_path, monkeypatch):
    import json

    import httpx

    settings = ServiceSettings(
        storage=tmp_path,
        api_key="synthetic-service-key-not-a-real-secret",
        urls=dict(chunking="http://chunking.invalid"),
    )
    app = create_service("pipeline", settings)
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer " + settings.api_key
    schema = app.openapi()
    assert "/v1/chunking/jobs" in schema["paths"]
    assert schema["paths"]["/v1/chunking/jobs"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ChunkingJob")
    assert "/v1/rag/runs" not in schema["paths"]
    assert (
        client.post("/v1/chunking/jobs", json=dict(source_artifact_id="../../.env")).status_code
        == 422
    )
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            202,
            json=dict(job_id="a" * 32, status="queued", status_url="/v1/jobs/" + "a" * 32),
            headers={"Location": "/v1/jobs/" + "a" * 32},
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), trust_env=False),
    )
    response = client.post("/v1/chunking/jobs", json=dict(source_artifact_id="b" * 32))
    assert response.status_code == 202
    assert response.json()["status_url"] == "/v1/chunking/jobs/" + "a" * 32
    assert response.headers["location"] == response.json()["status_url"]
    assert calls[0]["source_artifact_id"] == "b" * 32
