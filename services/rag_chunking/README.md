# Adaptive, source-linked parent/child preparation

**Parser choice:** quantized dots.mocr (Q4_K_M + Q8_0) is the production default. Original BF16 dots achieved our best selected handwritten-mathematics result (32/37), with **8.53 GiB peak PyTorch-allocated GPU memory** on a Colab T4. [Results, memory scope and limits](../../docs/evaluation/parser-model-selection.md).

Stage: `chunking`. Native development port: `8103`.

API contract: [openapi.json](openapi.json).
Shared implementation: `src/multimodal_doc_parser/pipeline_services/`.

```bash
document-pipeline-service api --service chunking --port 8103
document-pipeline-service worker --service chunking
```

See [service architecture](../../docs/architecture/service-pipeline.md) for authentication, artifact contracts, deployment and limitations.
