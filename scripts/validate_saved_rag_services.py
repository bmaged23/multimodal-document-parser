"""Live service smoke test starting from saved parsed data; no OCR rerun."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from multimodal_doc_parser.pipeline_services.service_runtime import import_run
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    previous = (
        json.loads((args.resume_from / "results.json").read_text()) if args.resume_from else None
    )
    settings = ServiceSettings.from_environment()
    destination = Path("outputs/quality_evaluations") / (
        "rag-microservices-smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    destination.mkdir()
    source = Path("outputs/parser_runs/original-final-source-checked-03")
    source_hash = hashlib.sha256((source / "document.json").read_bytes()).hexdigest()
    if previous and previous["source_document_sha256"] != source_hash:
        raise ValueError("Cannot reuse evidence from a different source document")
    artifact = previous["imported_artifact_id"] if previous else import_run(source, settings)
    records: dict[str, Any] = dict(
        source_run=str(source),
        source_document_sha256=source_hash,
        ocr_rerun=False,
        imported_artifact_id=artifact,
        stages=[],
        resumed_from=str(args.resume_from) if args.resume_from else None,
    )

    def save():
        (destination / "results.json").write_text(json.dumps(records, indent=2) + "\n")

    with httpx.Client(
        timeout=180, trust_env=False, headers={"Authorization": "Bearer " + settings.api_key}
    ) as client:

        def job(base, route, body):
            for earlier in previous.get("stages", []) if previous else []:
                if earlier["route"] == route and earlier["state"]["status"] == "completed":
                    check = client.get(base + route + "/" + earlier["job_id"])
                    check.raise_for_status()
                    state = check.json()
                    if state["status"] != "completed" or any(
                        state["request"].get(k) != v for k, v in body.items()
                    ):
                        raise ValueError("Saved service job is not compatible with this request")
                    records["stages"].append({**earlier, "reused": True})
                    save()
                    print("Reused " + route, flush=True)
                    return state["result"]
            response = client.post(
                base + route, json=body, headers={"Idempotency-Key": destination.name + route}
            )
            response.raise_for_status()
            created = response.json()
            print("Queued " + route, flush=True)
            started = time.monotonic()
            while time.monotonic() - started < 600:
                status = client.get(base + created["status_url"])
                status.raise_for_status()
                state = status.json()
                if state["status"] in ["completed", "failed", "interrupted"]:
                    records["stages"].append(
                        dict(
                            route=route,
                            job_id=created["job_id"],
                            state=state,
                            seconds=round(time.monotonic() - started, 2),
                        )
                    )
                    save()
                    if state["status"] != "completed":
                        raise ValueError("Service job did not complete: " + str(state.get("error")))
                    print("Completed " + route, flush=True)
                    return state["result"]
                time.sleep(1)
            raise TimeoutError("Service smoke job exceeded its deadline")

        try:
            chunks = job(
                "http://127.0.0.1:8100", "/v1/chunking/jobs", dict(source_artifact_id=artifact)
            )
            index = job(
                "http://127.0.0.1:8100",
                "/v1/indexing/jobs",
                dict(source_artifact_ids=[chunks["artifact_id"]], batch_size=2),
            )
            question = (
                "What are the mean latency in milliseconds and error rate "
                "in percent for sensor BY-04?"
            )
            if previous and previous.get("retrieval"):
                records["retrieval"] = previous["retrieval"]
            else:
                response = client.post(
                    "http://127.0.0.1:8105/v1/search",
                    json=dict(
                        index_artifact_id=index["artifact_id"], query=question, context_tokens=1800
                    ),
                )
                response.raise_for_status()
                records["retrieval"] = response.json()
            save()
            assert (
                records["retrieval"]["contexts"] and records["retrieval"]["context_tokens"] <= 1800
            )
            answer = job(
                "http://127.0.0.1:8106",
                "/v1/rag/runs",
                dict(
                    index_artifact_id=index["artifact_id"], question=question, context_tokens=1800
                ),
            )
            records["answer"] = answer
            records["selected_numeric_mentions_present"] = all(
                v in answer["answer"]["answer"] for v in ["23.1", "2.6"]
            )
            records["status"] = (
                "passed"
                if records["selected_numeric_mentions_present"]
                else "completed_requires_answer_review"
            )
        except Exception as exc:
            records["status"] = "failed"
            records["error"] = str(exc).replace(settings.api_key, "[REDACTED]")
            save()
            raise
    assert hashlib.sha256((source / "document.json").read_bytes()).hexdigest() == source_hash
    records["source_unchanged"] = True
    save()
    print("Saved integration evidence: " + str(destination), flush=True)


if __name__ == "__main__":
    main()
