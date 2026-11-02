# Milestones

Complete and verify each milestone before starting the next.

## Priority and showcase

The main deliverable is **parsing-model evaluation and comparison**, followed by
the final parsing pipeline. Present measured results, compelling source/output
previews and documented failures first. See the
[scope and showcase brief](project-scope-and-showcase.md).

Chunking, retrieval and the RAG agent are secondary supporting studies. Their
[completed comparison](../evaluation/simple-structural-versus-adaptive-rag.md)
is saved, including source-review caveats and frozen local artifacts. Keep this
work available without making further RAG expansion the default next milestone.
The next primary work is the parsing review/showcase, reproducible model/pipeline
evaluation, and release/deployment validation described in milestones 7–8.

The README showcase now includes actual public-source table/chart review
screenshots, a measured parser-comparison table and before/after pipeline checks.
Screenshot source hashes, attribution and capture details are saved under
`docs/assets/`. This completes the initial documentation showcase, not hosted
deployment or an independent benchmark.

## Current progress

Milestones 1–6 are implemented and verified locally within the scopes below;
the newly requested service architecture is also implemented. Docker deployment
and DOCX conversion are not locally verified. Milestone 1 established the uv
Python 3.12 environment, installable package, CI, documentation, and hardware report.

Milestone 2 adds:

- Versioned Pydantic models and an exported JSON Schema.
- Logical hierarchy, reading order, normalized source regions, table cells,
  visual asset references, and separate generated-content provenance.
- SHA-256 document identity and DPI-specific preparation bundles.
- PDF snapshots, serial page rendering, native resource cleanup, and atomic publication.
- Configurable page/pixel limits and rejection of invalid/encrypted inputs.
- Installed preparation/schema CLI commands and 20 automated tests.

Validation: locked sync, formatting, lint, strict source type checks, 20 tests,
source distribution, and wheel build passed. A local two-page user PDF was
prepared at 144 DPI; both 1190 × 1684 PNGs were verified and visually inspected.
This establishes preparation behavior, not OCR quality. The source PDF and
rendered assets remain local and ignored by Git.

At this milestone, CI had not run and the repository was still local.
Milestone 3 adds an isolated locked PaddleOCR runtime, saved-result adapter,
JSON/Markdown exports, visual crops, operational summaries, and HTML review with
the original PDF at left and extracted output at right. The local two-page PDF
ran successfully on the GTX 1660 Ti. Its raw results and diagnostics are preserved.
See `docs/evaluation/paddleocr-local-validation.md` for measurements and limitations.

Validation now includes 29 unit/integration checks plus a Chromium browser test
covering tabs, source highlights, page navigation, search, escaping, and responsive
layout. GPU runtime compatibility is confirmed for this input and configuration;
other backends and larger inputs remain unverified.

Post-milestone-3 improvements: separate rendered Markdown and Markdown source
views, embedded source crops in the preview, and a synchronized page-number
field with previous/next controls. A public-source evaluation corpus now contains
410 PDF pages across four documents, including historical handwritten scans.
The manifest pins checksums and identifies 16 development pages. These documents
have been downloaded and inspected, not scored by either backend yet.

Milestone 4 adds dots.mocr normalization, shared HTML-table handling, bounded
sequential HTTP inference, full response/input-image preservation, model and prompt
provenance, and comparisons restricted to matched source/preparation inputs.
Comparison JSON/HTML includes timings, sampled device memory, agreement, optional
reference CER/WER, and selected source-bound table-cell/span checks.

Quantized dots.mocr (Q4_K_M language + Q8_0 projector) ran twice on the GTX 1660 Ti
using a pinned CUDA 12.6/sm75 native build. Both two-page runs and their comparisons
are preserved in a separate model namespace. The official-prompt run took 33.77 s
and 24.09 s per page, with 5,901 MiB sampled device memory. Both parsers passed
20/20 source-derived cell checks on the simple sensor table. Chart/flowchart boxes
remain inaccurate; no general document accuracy claim is made. See
`docs/evaluation/dots-mocr-local-validation.md` for settings and limitations.

Validation includes 49 unit/integration/browser checks, Ruff, strict source type
checks, and package builds. At this milestone these were local results; subsequent hosted CI is recorded below.
The original, NASA-20, and final handwriting-5 PaddleOCR baselines are preserved;
The subsequent quantized GPU batch attempted NASA 20/20 and diary 5/5 pages,
accepting 20 and three respectively. Two diary pages reached the output token
limit. See [measured results](../evaluation/quantized-gpu-batch.md); broad accuracy
evaluation still requires independent annotations.

Milestone 5 adds visual enrichment through local Ollama, Ollama Cloud or optional
Qwen Cloud transports. The selected default is Gemma 4 31B on Ollama Cloud.
The original chart, flowchart and raster image completed; all 9 selected visual-type,
numeric and graph assertions passed, with manual review of the descriptions.
The heading-routing bug was removed. Model predictions, images/hashes, prompts,
native responses, provenance and failures are preserved. See
[Gemma validation](../evaluation/gemma4-cloud-visual-validation.md) for scope and limits.
This is local integration and a small development validation, not broad accuracy.

Pre-milestone-6 evaluation checkpoint completed: the 20-page IRENA selection and
26-page IMF appendix were parsed, reviewed, scored and preserved. Selected checks
passed 277/277 table-cell text assertions, 40/40 span assertions and 141/144 printed
chart-point assertions. Semantic header checks passed 0/32; one source photograph
was missed. See the [full evaluation](../evaluation/irena-imf-source-review.md)
for annotation limits, runtime/recovery and reproducible commands. The 76-test
local suite, Ruff and strict source typing passed. This is a single-assistant
development evaluation, not an independent model benchmark.

At the initial evaluation checkpoint, milestone 6 had not started. The user
requested a stop, received the evaluation summary and subsequently explicitly
authorized adaptive parent-child RAG implementation. The basic
review interface was brought forward from milestone 7 at the user's request.

Final repair pass: existing 48 OCR page checkpoints were replayed without GPU OCR.
The selected 277 cell-text, 40 span, 32 inferred-header and 144 chart-value checks
now pass after disclosed source postprocessing. The missing photograph is recovered,
table continuations/captions/notes are linked, and safe superscript rendering is
implemented. Raw Gemma results and source-corrected outputs remain separate.
See [final repairs](../evaluation/final-source-quality-repair.md).
The user's requested five-VLM/five-region comparison is prepared with diverse
line/bar/scatter charts and close-up/landscape photos. Gemma's cached cases are
scored; four alternatives return HTTP 402. It remains blocked by account access;
this is not a completed quality ranking. RAG had not started at that checkpoint.

The user selected Gemma 4 31B for continued cloud use with the second key.
The [selection checkpoint](../evaluation/gemma-selection-checkpoint.md) rechecked
the final 46-page source scores without OCR or cloud generations, retained raw
VLM error observations and recorded the access limitations of twelve alternative
hosted models. The current evaluation work is closed with documented limitations;
the comparative quality ranking remains unavailable. The user then authorized
milestone 6 and required separate microservice APIs.

Milestone 6 adds adaptive parent/child JSONL, exact evidence slices, explicit
context/exclusions, table continuation and column-group preservation, token
budgets, validation and a source/parent/child HTML inspector. The saved 48 pages
produce 88 parents and 692 searchable children. BGE-large-en-v1.5 generated
693 embedding windows on CUDA with exactly 1,024 dimensions; BM25/dense rank
fusion and bounded parent expansion are implemented.

The requested architecture adds independent document preparation, parsing,
chunking, indexing and retrieval APIs/workers, plus separate public pipeline and
agentic RAG APIs. Jobs persist with idempotency and worker heartbeats; services
exchange artifact IDs and coordinate single-host GPU use. A live saved-output
test completed chunking, GPU indexing/retrieval and a cited cloud Gemma answer
without rerunning OCR. Three small retrieval questions pass the selected source
location checks; they do not establish best chunking or general answer accuracy.

See [RAG validation](../evaluation/adaptive-rag-development.md) and
[service architecture](../architecture/service-pipeline.md). Container definitions
and CI build validation are configured. The parsing container passed API/auth,
DOCX conversion, page rendering and GPU CUDA checks. The public Docker Hub
release is published and anonymous registry access verified. GitHub CI runs on source
pushes; the Docker Hub release environment is configured. Held-out retrieval/answer
evaluation is recorded separately; a multi-model VLM winner remains unavailable.

## Delivery sequence

1. **Foundation and hardware:** uv environment, project configuration, setup
   documentation, basic CI, package build, and GPU/runtime compatibility findings.
2. **Schema and PDF preparation:** shared document schema, page rendering,
   document IDs, coordinates, and asset storage.
3. **PaddleOCR:** adapter, raw results, normalized JSON, and Markdown export.
4. **dots.mocr and comparison:** second adapter and measured quality, latency,
   and GPU memory comparisons using the same document inputs.
5. **Visual understanding:** Ollama vision integration, chart values and image
   descriptions, source crops, and explicit value provenance.
6. **Secondary RAG preparation:** structure-aware chunking and JSONL with source
   metadata; downstream comparisons are preserved as supporting experiments.
7. **Parsing showcase, review and deployment:** lead with model evaluation,
   source/output visualizations and parser differences; add Docker,
   a deployed example explorer, and documented GPU deployment.
8. **Evaluation and release:** varied annotated examples, meaningful checks,
   reproducible benchmark results, limitations, release image publishing,
   and GitHub documentation.

The milestone 7 saved-results explorer is implemented and browser-checked locally.
It includes expanded/resizable PDF and output panes, zoom, comparison tabs,
source regions and public-source examples. Versioned Docker Hub release automation
and parsing-only Compose deployment are defined. Local image smoke checks passed,
including DOCX conversion and CUDA allocation. The public Docker Hub release
is published with anonymous manifest verification, SBOM and provenance.
The public GitHub repository is connected and CI runs on source pushes.
Python/browser CI passed, and the Docker Hub environment token is configured.
Hosted API/parsing image checks and the first automated Docker Hub release
(`0.1.0-rc.2`) passed. The 53-page, four-document public saved-output demo is deployed on
GitHub Pages and browser-verified; a live hosted inference service remains separate.
The GitHub Release contains the image record, portable demo, manifest and checksums.
