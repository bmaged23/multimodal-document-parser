# Task scripts

Run scripts from the repository root through the existing uv environment.
Core application behavior belongs in `src/multimodal_doc_parser/`; scripts
orchestrate setup, repeatable experiments, deployment checks and presentations.

| Folder | Purpose |
| --- | --- |
| [model_runtimes/](model_runtimes/) | Model download, native GPU build and server startup |
| [document_processing/](document_processing/) | Corpus download and controlled parsing resume |
| [parser_evaluation/](parser_evaluation/) | Parser batches, source review and immutable run preservation |
| [visual_evaluation/](visual_evaluation/) | Chart/image description benchmarks and selected-source scoring |
| [rag_evaluation/](rag_evaluation/) | Question/reference generation, retrieval, answers and judge audits |
| [service_management/](service_management/) | Local secrets, service launch, contracts and saved-run validation |
| [deployment/](deployment/) | Container smoke checks and guarded hosted-runner preparation |
| [visualization/](visualization/) | Saved-output explorers, review pages and screenshot checks |

## Commands and responsibilities

### Model Runtimes

| Script | Responsibility |
| --- | --- |
| [build_dots_mocr_runtime.py](model_runtimes/build_dots_mocr_runtime.py) | Build pinned llama.cpp for the GTX 1660 Ti using an isolated CUDA 12.6 toolchain. |
| [serve_dots_mocr.py](model_runtimes/serve_dots_mocr.py) | Serve quantized dots.mocr on GPU through the isolated native runtime. |
| [setup_dots_mocr_runtime.py](model_runtimes/setup_dots_mocr_runtime.py) | Download checksum-pinned dots.mocr GGUF weights and an isolated llama.cpp runtime. |
| [setup_ollama_runtime.py](model_runtimes/setup_ollama_runtime.py) | Install a checksummed portable Ollama release in the ignored project cache. |
| [setup_unlimited_ocr_2_runtime.py](model_runtimes/setup_unlimited_ocr_2_runtime.py) | Download the pinned official Unlimited-OCR 2 snapshot, verifying LFS hashes. |

### Document Processing

| Script | Responsibility |
| --- | --- |
| [download_evaluation_documents.py](document_processing/download_evaluation_documents.py) | Download manifest-listed PDF sources locally, preserving reproducible checksums. |
| [resume_parsing_run.py](document_processing/resume_parsing_run.py) | Resume an interrupted single-model run and refresh a clearly marked live report. |

### Parser Evaluation

| Script | Responsibility |
| --- | --- |
| [archive_evaluation_runs.py](parser_evaluation/archive_evaluation_runs.py) | Create immutable, checksum-inventoried evaluation snapshots and watch final exports. |
| [evaluate_handwritten_math_comparison.py](parser_evaluation/evaluate_handwritten_math_comparison.py) | Score frozen selected handwriting assertions and ordered truth-table cells; save per-check results. |
| [run_handwritten_math_comparison.py](parser_evaluation/run_handwritten_math_comparison.py) | Prepare four author-provided photographs and run the three retained GPU profiles sequentially. |
| [evaluate_source_review.py](parser_evaluation/evaluate_source_review.py) | Reproduce source-reviewed quality checks against a saved parser run; no inference. |
| [finalize_saved_visual_quality.py](parser_evaluation/finalize_saved_visual_quality.py) | Apply source-only quality checks to saved model responses without new inference. |
| [run_final_quality_repair.py](parser_evaluation/run_final_quality_repair.py) | Replay saved OCR and run one bounded visual repair batch; never launch GPU OCR. |
| [run_quantized_evaluation_batch.py](parser_evaluation/run_quantized_evaluation_batch.py) | Evaluate quantized dots.mocr page by page, preserving successes and failures. |
| [run_unlimited_ocr_2_evaluation_batch.py](parser_evaluation/run_unlimited_ocr_2_evaluation_batch.py) | Run the preserved NASA-20 and diary-5 selections with bounded GPU page jobs. |
| [run_unlimited_ocr_2_validation.py](parser_evaluation/run_unlimited_ocr_2_validation.py) | Wait for staged downloads, run a bounded GPU probe, preserve diagnostic artifacts. |

### Visual Evaluation

| Script | Responsibility |
| --- | --- |
| [evaluate_visual_enrichment.py](visual_evaluation/evaluate_visual_enrichment.py) | Score selected source-derived chart and flowchart assertions, not global accuracy. |
| [run_visual_enrichment_validation.py](visual_evaluation/run_visual_enrichment_validation.py) | Wait for local model setup, enrich the original PDF, and preserve its evaluation. |
| [run_vlm_benchmark.py](visual_evaluation/run_vlm_benchmark.py) | Compare raw VLM responses on fixed saved crops; never rerun OCR or repair values. |

### Rag Evaluation

| Script | Responsibility |
| --- | --- |
| [audit_rag_evaluation.py](rag_evaluation/audit_rag_evaluation.py) | Record an assistant review of 20 saved cases, without re-parsing or re-answering. |
| [build_rag_benchmark_100.py](rag_evaluation/build_rag_benchmark_100.py) | Build source-bound questions and freeze group-separated 30/70 development/test splits. |
| [compare_rag_approaches.py](rag_evaluation/compare_rag_approaches.py) | Controlled chunk/retrieval/RAG ablation with checkpointed, paired results. |
| [compare_structural_rag_baseline.py](rag_evaluation/compare_structural_rag_baseline.py) | Compare practical structural plain RAG with the existing adaptive/agent results. |
| [evaluate_rag_retrieval.py](rag_evaluation/evaluate_rag_retrieval.py) | Small source-derived retrieval checks; no answer generation or OCR calls. |
| [generate_rag_reference_drafts.py](rag_evaluation/generate_rag_reference_drafts.py) | Preserve GPT-OSS reference drafts separately from source-compiled fact targets. |
| [report_rag_retrieval_comparison.py](rag_evaluation/report_rag_retrieval_comparison.py) | Report bounded source-cell coverage independently of any LLM judge. |
| [run_ragas_benchmark.py](rag_evaluation/run_ragas_benchmark.py) | Checkpointed retrieval/oracle/agent evaluation; persist every score and judge call. |
| [validate_rag_judge.py](rag_evaluation/validate_rag_judge.py) | Validate the evaluator on preregistered synthetic controls, not benchmark tuning. |

### Service Management

| Script | Responsibility |
| --- | --- |
| [configure_service_secrets.py](service_management/configure_service_secrets.py) | Create ignored local service credentials without printing or rotating existing keys. |
| [export_service_contracts.py](service_management/export_service_contracts.py) | Export each independently deployable service's OpenAPI contract without secrets. |
| [run_development_services.py](service_management/run_development_services.py) | Start or stop separate local service processes using the existing uv environment. |
| [validate_saved_rag_services.py](service_management/validate_saved_rag_services.py) | Live service smoke test starting from saved parsed data; no OCR rerun. |

### Deployment

| Script | Responsibility |
| --- | --- |
| [prepare_hosted_container_runner.sh](deployment/prepare_hosted_container_runner.sh) | Reclaim unused toolchains only on ephemeral GitHub-hosted Linux runners. |
| [verify_container_image.py](deployment/verify_container_image.py) | Smoke-test the distributable parsing image without inference or credentials. |

### Visualization

| Script | Responsibility |
| --- | --- |
| [build_evaluation_showcase.py](visualization/build_evaluation_showcase.py) | Build the read-only parser/pipeline showcase from saved runs, without inference. |
| [build_public_demo.py](visualization/build_public_demo.py) | Create allowlisted public excerpts from checksum-verified saved results. |
| [render_parsing_pipeline_animation.py](visualization/render_parsing_pipeline_animation.py) | Draw the implementation-checked animated pipeline and demo banner. |
| [capture_public_demo_gallery.py](visualization/capture_public_demo_gallery.py) | Capture five real output previews with source attribution. |
| [build_unlimited_ocr_2_review.py](visualization/build_unlimited_ocr_2_review.py) | Create a diagnostic PDF/Markdown review without claiming canonical normalization. |
| [capture_review_preview.py](visualization/capture_review_preview.py) | Capture a report preview using the optional browser dependency group. |
| [verify_saved_review_reports.py](visualization/verify_saved_review_reports.py) | Verify real saved HTML reports in Chromium; uses the optional browser uv group. |

## Execution and preserved evidence

Use `uv run --locked python scripts/<group>/<script>.py --help` for scripts with
an argument parser; some experiment runners are fixed research protocols, so read
their source and evaluation guide before executing them. Scripts may start GPU
inference or send cloud requests. Merely installing the project runs no benchmark.

The previous flat `scripts/<name>` paths moved into these folders. Active source,
tests, workflows and documentation use the new paths. Historical archive manifests
keep their original source paths and hashes: frozen evidence is not rewritten.
Private PDFs, model weights and `outputs/evaluation_archive/` remain in their
existing ignored locations.
