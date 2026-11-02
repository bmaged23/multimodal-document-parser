"""Export each independently deployable service's OpenAPI contract without secrets."""

import json
import tempfile
from pathlib import Path

from multimodal_doc_parser.pipeline_services.service_api import create_service
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings

SERVICES = {
    "document_pipeline_api": ("pipeline", 8100, "Public ingestion orchestrator and stage facade"),
    "document_preparation": ("documents", 8101, "PDF/DOCX upload and PDF rendering"),
    "document_parsing": ("parsing", 8102, "GPU OCR and optional cloud visual enrichment"),
    "rag_chunking": ("chunking", 8103, "Adaptive, source-linked parent/child preparation"),
    "embedding_indexing": ("indexing", 8104, "1024-dimensional BGE-large GPU indexing and BM25"),
    "retrieval": ("retrieval", 8105, "Hybrid retrieval and bounded parent expansion"),
    "agentic_rag_api": ("rag", 8106, "Public bounded agentic question-answering API"),
}


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        settings = ServiceSettings(
            storage=Path(temporary), api_key="documentation-only-placeholder-not-a-secret"
        )
        for folder, (stage, port, purpose) in SERVICES.items():
            destination = Path("services") / folder
            destination.mkdir(parents=True, exist_ok=True)
            app = create_service(stage, settings)
            (destination / "openapi.json").write_text(json.dumps(app.openapi(), indent=2) + "\n")
            (destination / "README.md").write_text(
                f"# {purpose}\n\nStage: `{stage}`. Native development port: `{port}`.\n\n"
                "API contract: [openapi.json](openapi.json).\n"
                "Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.\n\n"
                f"```bash\ndocument-pipeline-service api --service {stage} --port {port}\n"
                + (
                    f"document-pipeline-service worker --service {stage}\n"
                    if stage != "retrieval"
                    else ""
                )
                + "```\n\nSee [service architecture](../../docs/architecture/service-pipeline.md) "
                "for authentication, artifact contracts, deployment and limitations.\n"
            )
    print("Exported seven service API contracts; no credentials included.")


if __name__ == "__main__":
    main()
