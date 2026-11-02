# Adaptive, source-linked parent/child preparation

Stage: `chunking`. Native development port: `8103`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service chunking --port 8103
document-pipeline-service worker --service chunking
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
