# Public bounded agentic question-answering API

Stage: `rag`. Native development port: `8106`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service rag --port 8106
document-pipeline-service worker --service rag
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
