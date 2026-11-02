# Project documentation

Start with the [main README](../README.md) for visual examples and comparison
results. The documentation below follows the project's priorities.

| Guide | Purpose | Priority |
| --- | --- | --- |
| [Parsing evaluation and model comparison](parsing-evaluation/README.md) | Models, cases, metrics, results, failures and reproducibility | Primary |
| [Parsing pipeline](parsing-pipeline/README.md) | Preparation, OCR, normalization, visual enrichment, exports and review | Primary |
| [Chunking, retrieval and RAG](chunking-and-rag/README.md) | Downstream approaches, completed comparison, agent and judge limitations | Secondary |

The guides distinguish what is implemented, what was measured, and what remains
unverified. Detailed experiment reports are linked from each guide rather than
presented as one undifferentiated benchmark.

- [Interactive saved-results explorer](development/evaluation-showcase.md)
- [Docker image, public release and CI/CD](deployment/docker-hub.md)
- [Project scope and showcase](project/project-scope-and-showcase.md)
- [Milestones](project/milestones.md)
- [Hardware findings](development/hardware-report.md)
- [Document schema](reference/document-schema.md)
- [Service architecture](architecture/service-pipeline.md)
- [Preserved artifacts](evaluation/preserved-artifacts.md)

Private PDFs, raw predictions, full local question/evidence records, model weights
and credentials are excluded from Git and package distributions. Curated
public-source screenshots and non-private measured summaries remain in the
repository. See [screenshot provenance](assets/showcase-provenance.json).
