# GPU OCR and optional cloud visual enrichment

Stage: `parsing`. Native development port: `8102`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service parsing --port 8102
document-pipeline-service worker --service parsing
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
