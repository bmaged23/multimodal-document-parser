# 1024-dimensional BGE-large GPU indexing and BM25

Stage: `indexing`. Native development port: `8104`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service indexing --port 8104
document-pipeline-service worker --service indexing
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
