# Adaptive RAG and microservice development validation

Implemented on 2026-10-31 using the saved final parser outputs. **No OCR rerun**
was needed. The new stages are independent of the parser's model runtime and
preserve the original JSON, PDFs, raw VLM responses and evaluation archives.

## Chunking and GPU indexing

| Saved case | Pages | Parents | Searchable children | Largest child, cl100k tokens |
| --- | --- | --- | --- | --- |
| Original multimodal sample | 2 | 13 | 13 | 256 |
| IRENA mixed content | 20 | 52 | 95 | 350 |
| IMF complex tables | 26 | 23 | 584 | 465 |
| Total | 48 | 88 | 692 | 465 |

All body slices, table-cell references and source elements are accounted for.
Page furniture and empty elements are explicit exclusions: 2 original, 36 IRENA
and 93 IMF elements. Source coverage is not an OCR accuracy measurement. The
canonical parser's generated descriptions and inferred headers remain labelled.

Ten wide IMF table-header records did not fit next to rows in the initial
350/500-token configuration. They now have independent searchable children;
context is never silently dropped from search. Existing data rows retain their
full header paths and exact cell coordinates. Continued tables share logical
parents but different column layouts do not share a child. The first development
bundles/index remain separate from the final `02` versions.

The model is **BAAI/bge-large-en-v1.5**, using Qdrant's ONNX export at revision
`e93b9305e013fdf771b471426516285da0cca188`. It produced **1,024-dimensional**
L2-normalized CLS vectors using a CUDA session. An actual execution profile
confirms CUDA nodes; CPU shape/control nodes are permitted. The 692 children
required **693 windows**, individually bounded by BGE's own tokenizer. Text tails
are retained instead of silently truncated. Batch size was two on the 6 GB GPU.
Exact model/runtime and file hashes are in the index manifest.

Local artifacts:

- `outputs/rag_bundles/original-adaptive-parent-child-02/`
- `outputs/rag_bundles/irena-adaptive-parent-child-02/`
- `outputs/rag_bundles/imf-adaptive-parent-child-02/`
- `outputs/rag_indexes/bge-large-hybrid-02/`

Each bundle contains parent/child JSONL, a manifest, validation results and an
HTML source/child/parent inspector. Parent and child JSON Schemas are published
under `schemas/`. The inspector uses body evidence for page filtering, rather
than including every page containing a repeated heading or footnote.

## Retrieval development checks

Three source-derived questions target a sensor table, IRENA's technology-cost
comparison table and IMF's oil-price assumptions paragraph. Each was run with
keyword-only/hybrid search and with/without parent expansion, at top-k 5 and the
same 2,400-token formatted context budget.

All 12 runs retrieved the selected source element and retained it in the final
context. No context exceeded the configured budget. **These are three easy
development questions, not 12 independent benchmark questions.** Both retrieval
modes passed; this does not demonstrate a hybrid or parent-expansion advantage.
There is no held-out fixed-size chunking comparison yet.

Evidence: `outputs/quality_evaluations/adaptive-rag-retrieval-01/results.json`.
Reproduce using `scripts/rag_evaluation/evaluate_rag_retrieval.py` with a fresh output directory.

## Service and answer checks

Seven independent API processes and six workers were exercised natively. The
public pipeline and agentic RAG APIs have different ports and OpenAPI contracts.
The RAG contract contains no upload/ingestion endpoints. Stage APIs use artifact
IDs; invalid path-like IDs fail validation. Authenticated, typed pipeline routes
forward job creation/status and preserve their public status links.

API tests cover authentication, upload/preparation, idempotency replay/conflict,
exclusive job claims, expired-worker interruption without automatic retry,
gateway request contracts and separation of public APIs. Agent tests cover
bounded rewriting, fenced JSON and rejection of invented citation identifiers.

The live saved-output smoke test imported the original reviewed parser result,
completed chunking through the public pipeline API, completed GPU indexing in
the separate worker, retrieved through the retrieval API and generated an answer
through the agentic RAG API. Cloud Gemma correctly answered BY-04's mean latency
as **23.1 ms** and error rate as **2.6%**, with a supplied child evidence ID.
The original parser JSON hash remained unchanged.

The first agent attempt exposed a complete Markdown JSON fence. The transport
now unwraps only that complete envelope, without repairing JSON or extracting it
from arbitrary prose. Its failed trace is retained. The successful explicit
retry reused the completed chunking/indexing jobs and saved retrieval probe.
No OCR or embedding job was repeated for the retry.

Successful live evidence:
`outputs/quality_evaluations/rag-microservices-smoke-20261031T193109Z/results.json`.
The answer artifact retains the retrieved evidence, model decisions, native
responses and raw prompt trace. Citation validation checks referential integrity;
independent entailment and broad answer quality are not established by one case.

## Software checks and limits

The full local suite passed **108 tests**, including browser, API and worker
checks. Ruff and strict typing passed for 43 source files. Source/child/parent
inspection was also exercised on the real saved cases. Hosted CI remains unrun.

Compose defines separate APIs, workers, GPU images and a LibreOffice conversion
image. Only the two public API ports are published by default. GPU build targets,
service network exposure and seven OpenAPI documents were checked structurally.
**Container builds/deployment are unverified:** Docker Desktop WSL integration
is unavailable. DOCX conversion also remains unverified on this host; it requires
the documents image or LibreOffice. The existing parser was not rerun through
the new orchestration service.

This is a single-host development system with shared filesystem artifacts and
per-stage SQLite queues. It is not a multi-host or multi-tenant production
deployment. New documents, noisy handwriting, unanswerable questions, footnote
dependencies, cross-page reasoning and adversarial document instructions need
broader independent evaluation. An oversized individual source part is split
losslessly with offsets and a warning; a general cell-aware partitioner for
arbitrarily oversized individual table rows is not implemented.

See [architecture and operating instructions](../architecture/service-pipeline.md).
