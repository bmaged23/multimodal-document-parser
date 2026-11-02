"""Stage-specific FastAPI applications; the RAG API does not expose ingestion."""

import asyncio
import hashlib
import secrets
import shutil
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from multimodal_doc_parser.pipeline_services.contracts import JOB_REQUESTS, SearchRequest
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.storage import ArtifactStore, JobStore


def create_service(stage: str, settings: ServiceSettings | None = None) -> FastAPI:
    if stage not in {*JOB_REQUESTS, "retrieval"}:
        raise ValueError("Unknown service stage")
    settings = settings or ServiceSettings.from_environment()
    artifacts = ArtifactStore(settings.storage)
    jobs = JobStore(settings.storage, stage)
    app = FastAPI(
        title=f"Document platform: {stage}",
        version="1.0.0",
        dependencies=[Depends(HTTPBearer(auto_error=False))],
        description=(
            "Bearer-authenticated stage API. Durable jobs return 202; artifact IDs "
            "reference completed immutable outputs."
        ),
    )
    app.state.jobs, app.state.artifacts = jobs, artifacts

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if request.url.path not in {
            "/health/live",
            "/health/ready",
            "/docs",
            "/openapi.json",
            "/redoc",
        }:
            provided = request.headers.get("authorization", "")
            if not secrets.compare_digest(provided, "Bearer " + settings.api_key):
                return JSONResponse(
                    dict(detail="Valid service bearer authentication required"), status_code=401
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return dict(status="live", service=stage)

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        with jobs.connect() as db:
            db.execute("SELECT 1")
        return dict(
            status="ready", service=stage, api_version="1.0", worker_required=stage != "retrieval"
        )

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact_metadata(artifact_id: str) -> dict[str, Any]:
        try:
            result = artifacts.metadata(artifact_id)
            return {k: v for k, v in result.items() if k != "bundle"}
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "Unknown completed artifact") from exc

    @app.get("/v1/artifacts/{artifact_id}/files/{filename:path}")
    def artifact_file(artifact_id: str, filename: str) -> FileResponse:
        try:
            root = artifacts.resolve(artifact_id).resolve()
            path = (root / filename).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or any(p.startswith(".") for p in Path(filename).parts)
            ):
                raise ValueError("Invalid artifact file")
            return FileResponse(path)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "Unknown artifact file") from exc

    if stage != "retrieval":
        status_base = (
            "/v1/rag/runs"
            if stage == "rag"
            else ("/v1/pipelines" if stage == "pipeline" else "/v1/jobs")
        )

        async def submit(body: BaseModel, request: Request) -> JSONResponse:
            key = request.headers.get("idempotency-key")
            if key is not None and (not key.strip() or len(key) > 200):
                raise HTTPException(400, "Idempotency-Key must contain 1–200 characters")
            try:
                record, created = jobs.enqueue(body.model_dump(mode="json"), key)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            location = status_base + "/" + record["id"]
            return JSONResponse(
                dict(job_id=record["id"], status=record["status"], status_url=location),
                status_code=202 if created else 200,
                headers={"Location": location},
            )

        submit.__annotations__["body"] = JOB_REQUESTS[stage]
        app.add_api_route(
            status_base,
            submit,
            methods=["POST"],
            status_code=202,
        )

        @app.get(status_base + "/{job_id}")
        def job_status(job_id: str) -> dict[str, Any]:
            try:
                record = jobs.get(job_id)
                return {
                    k: v
                    for k, v in record.items()
                    if k not in {"request_hash", "idempotency_key", "worker"}
                }
            except KeyError as exc:
                raise HTTPException(404, "Unknown job") from exc

    if stage == "documents":

        @app.post("/v1/documents", status_code=201)
        async def upload(file: UploadFile) -> dict[str, Any]:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in {".pdf", ".docx"}:
                raise HTTPException(415, "Upload a PDF or DOCX file")
            identity, directory = artifacts.reserve()
            try:
                path = directory / ("upload" + suffix)
                digest = hashlib.sha256()
                size = 0
                with path.open("xb") as stream:
                    while block := await file.read(1024 * 1024):
                        size += len(block)
                        if size > 100 * 1024 * 1024:
                            raise HTTPException(413, "Upload exceeds 100 MiB")
                        digest.update(block)
                        stream.write(block)
                if size == 0:
                    raise HTTPException(422, "Empty upload")
                with path.open("rb") as stream:
                    magic = stream.read(8)
                if (suffix == ".pdf" and not magic.startswith(b"%PDF-")) or (
                    suffix == ".docx" and not magic.startswith(b"PK")
                ):
                    raise HTTPException(422, "File contents do not match the extension")
                return artifacts.publish(
                    identity,
                    "upload",
                    directory,
                    filename=path.name,
                    display_filename=Path(file.filename or "document").name,
                    sha256=digest.hexdigest(),
                    bytes=size,
                )
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            finally:
                await file.close()

    if stage == "retrieval":

        @app.post("/v1/search")
        async def search(body: SearchRequest) -> dict[str, Any]:
            from multimodal_doc_parser.pipeline_services.stage_handlers import search_index

            try:
                return await asyncio.to_thread(search_index, body, settings)
            except (OSError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc

    if stage == "pipeline":

        async def proxy(service: str, path: str, request: Request) -> Response:
            import httpx

            if (
                service not in {"preparation", "parsing", "chunking", "indexing"}
                or ".." in Path(path).parts
            ):
                raise HTTPException(404, "Unknown pipeline stage")
            domain = service
            service = "documents" if service == "preparation" else service
            headers = {
                name: request.headers[name]
                for name in ["authorization", "content-type", "idempotency-key"]
                if name in request.headers
            }
            async with httpx.AsyncClient(
                timeout=120, trust_env=False, follow_redirects=False
            ) as client:
                try:
                    response = await client.request(
                        request.method,
                        settings.urls[service] + "/v1/" + path,
                        headers=headers,
                        content=request.stream(),
                        params=request.query_params,
                    )
                except httpx.HTTPError as exc:
                    raise HTTPException(503, "Stage service unavailable") from exc
            forwarded = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in {"content-type", "location"}
            }
            if "location" in forwarded:
                forwarded["location"] = forwarded["location"].replace("/v1/", f"/v1/{domain}/", 1)
            if "application/json" in response.headers.get("content-type", ""):
                data = response.json()
                if isinstance(data, dict) and isinstance(data.get("status_url"), str):
                    data["status_url"] = data["status_url"].replace("/v1/", f"/v1/{domain}/", 1)
                    return JSONResponse(data, response.status_code, headers=forwarded)
            return Response(response.content, response.status_code, headers=forwarded)

        def register_stage(domain: str, service: str) -> None:
            async def create_job(body: BaseModel, request: Request) -> Response:
                return await proxy(domain, "jobs", request)

            create_job.__annotations__["body"] = JOB_REQUESTS[service]
            app.add_api_route(f"/v1/{domain}/jobs", create_job, methods=["POST"], status_code=202)

            async def read_job(job_id: str, request: Request) -> Response:
                return await proxy(domain, "jobs/" + job_id, request)

            app.add_api_route(f"/v1/{domain}/jobs/{{job_id}}", read_job, methods=["GET"])

        for domain, service in [
            ("preparation", "documents"),
            ("parsing", "parsing"),
            ("chunking", "chunking"),
            ("indexing", "indexing"),
        ]:
            register_stage(domain, service)

        @app.post(
            "/v1/documents",
            status_code=201,
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["file"],
                                "properties": {"file": {"type": "string", "format": "binary"}},
                            }
                        }
                    },
                }
            },
        )
        async def upload_proxy(request: Request) -> Response:
            import httpx

            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                try:
                    response = await client.post(
                        settings.urls["documents"] + "/v1/documents",
                        headers={
                            "Authorization": "Bearer " + settings.api_key,
                            "Content-Type": request.headers.get("content-type", ""),
                        },
                        content=request.stream(),
                    )
                except httpx.HTTPError as exc:
                    raise HTTPException(503, "Documents service unavailable") from exc
            return Response(response.content, response.status_code, media_type="application/json")

    return app
