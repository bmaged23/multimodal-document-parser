# Project documentation

**Model choice:** dots.mocr is our preferred model family overall on the 70-page test corpus. Original BF16 offers the strongest overall balance; **Q4_K_M language + Q8_0 projector** is the production default for the **GTX 1660 Ti with 6 GB VRAM**. BF16 reached **8.53 GiB peak PyTorch allocation** on Colab T4. [Comparison, measured memory and limits](evaluation/parser-model-selection.md).

Start with the [main README](../README.md) for visual examples and comparison
results. The documentation below follows the project's priorities.

| Guide | Purpose | Priority |
| --- | --- | --- |
| [70-page evaluation methodology and report](evaluation/README.md) | Source references, metric definitions, ranking and saved results | Primary |
| [Parsing evaluation and model comparison](parsing-evaluation/README.md) | Models, cases, metrics, results, failures and reproducibility | Primary |
| [Parsing pipeline](parsing-pipeline/README.md) | Preparation, OCR, normalization, visual enrichment, exports and review | Primary |
| [Chunking, retrieval and RAG](chunking-and-rag/README.md) | Downstream approaches, completed comparison, agent and judge limitations | Secondary |

The guides distinguish what is implemented, what was measured, and what remains
unverified. Detailed experiment reports are linked from each guide rather than
presented as one undifferentiated benchmark.

- [Live saved-output demo](https://bmaged23.github.io/multimodal-document-parser/)
- [Four-model demo design, attribution and reproduction](demo/README.md)
- [Hosted demo verification](../evaluation/results/public-demo-validation-v5.1.json)
- [Interactive explorer and local reproduction](development/evaluation-showcase.md)
- [Unlimited-OCR 2 GPU evaluation](evaluation/unlimited-ocr-2-gpu-results.md)
- [API setup and request examples](api/README.md)
- [Docker clone, pull and deployment](docker/README.md)
- [CI/CD checks and release automation](ci-cd/README.md)
- [Script directory guide](../scripts/README.md)
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

- [Author handwritten mathematics comparison](evaluation/author-handwritten-mathematics.md)

- [Four-profile evaluation across all 31 shared pages](evaluation/parser-corpus-four-profiles.md)
- [Model selection and measured GPU memory](evaluation/parser-model-selection.md)
