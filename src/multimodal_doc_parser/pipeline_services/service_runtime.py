"""Run one API or durable stage worker per process."""

import argparse
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from multimodal_doc_parser.pipeline_services.contracts import JOB_REQUESTS
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.storage import ArtifactStore, JobStore


def work_once(stage: str, settings: ServiceSettings) -> bool:
    from multimodal_doc_parser.pipeline_services.stage_handlers import execute_stage

    jobs = JobStore(settings.storage, stage)
    job = jobs.claim(str(os.getpid()) + ":" + uuid.uuid4().hex)
    if job is None:
        return False
    stop = threading.Event()

    def pulse() -> None:
        while not stop.wait(10):
            jobs.heartbeat(job["id"])

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    try:
        result = execute_stage(stage, job["request"], settings)
        jobs.finish(job["id"], result=result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}".replace(settings.api_key, "[REDACTED]")
        ollama_key = os.environ.get("OLLAMA_API_KEY")
        if ollama_key:
            error = error.replace(ollama_key, "[REDACTED]")
        jobs.finish(job["id"], error=error[:1000])
    finally:
        stop.set()
        heartbeat.join(timeout=2)
    return True


def import_run(source: Path, settings: ServiceSettings, kind: str = "parsed") -> str:
    from multimodal_doc_parser.document_schema.document import Document

    store = ArtifactStore(settings.storage)
    identity, directory = store.reserve()
    target = directory / "bundle"
    target.mkdir()
    if kind in {"parsed", "prepared"}:
        doc = Document.model_validate_json((source / "document.json").read_text())
        if kind == "parsed" and doc.status != "parsed":
            raise ValueError("Import requires a completed parser run")
        paths = {"document.json", "source.pdf"}
        paths.update(e.asset.path for e in doc.elements if hasattr(e, "asset"))
        paths.update(p.image.path for p in doc.pages)
        for relative in paths:
            origin = (source / relative).resolve()
            if not origin.is_relative_to(source.resolve()):
                raise ValueError("Unsafe source asset")
            if origin.exists():
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
    elif kind in {"chunks", "index"}:
        # Copy only the completed bundle, never model caches or environments.
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        raise ValueError("Unsupported import kind")
    store.publish(identity, kind, target, imported_from=str(source))
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    api = commands.add_parser("api")
    api.add_argument("--service", choices=[*JOB_REQUESTS, "retrieval"], required=True)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8100)
    worker = commands.add_parser("worker")
    worker.add_argument("--service", choices=list(JOB_REQUESTS), required=True)
    worker.add_argument("--once", action="store_true")
    importer = commands.add_parser(
        "import-run", help="Owner-only CLI import of saved outputs; no inference"
    )
    importer.add_argument("source", type=Path)
    importer.add_argument(
        "--kind", choices=["parsed", "prepared", "chunks", "index"], default="parsed"
    )
    args = parser.parse_args()
    settings = ServiceSettings.from_environment()
    if args.command == "api":
        import uvicorn

        from multimodal_doc_parser.pipeline_services.service_api import create_service

        uvicorn.run(create_service(args.service, settings), host=args.host, port=args.port)
    elif args.command == "import-run":
        print(json.dumps(dict(artifact_id=import_run(args.source, settings, args.kind))))
    else:
        while True:
            ran = work_once(args.service, settings)
            if args.once:
                break
            if not ran:
                time.sleep(1)


if __name__ == "__main__":
    main()
