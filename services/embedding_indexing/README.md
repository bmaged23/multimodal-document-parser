# 1024-dimensional BGE-large GPU indexing and BM25

**Parser choice:** quantized dots.mocr (Q4_K_M + Q8_0) is the production default. Original BF16 dots achieved our best selected handwritten-mathematics result (32/37), with **8.53 GiB peak PyTorch-allocated GPU memory** on a Colab T4. [Results, memory scope and limits](../../docs/evaluation/parser-model-selection.md).

Stage: `indexing`. Native development port: `8104`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service indexing --port 8104
document-pipeline-service worker --service indexing
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
