# Public ingestion orchestrator and stage facade

Stage: `pipeline`. Native development port: `8100`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service pipeline --port 8100
document-pipeline-service worker --service pipeline
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
