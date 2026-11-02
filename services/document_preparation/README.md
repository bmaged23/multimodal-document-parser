# PDF/DOCX upload and PDF rendering

Stage: `documents`. Native development port: `8101`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service documents --port 8101
document-pipeline-service worker --service documents
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
