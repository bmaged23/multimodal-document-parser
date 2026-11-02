# Project priorities and presentation

The primary purpose is evaluating and comparing document parsing models and the
final parsing pipeline. Lead the project showcase with measured parser results,
source/output visualizations and documented failures. Distinguish raw model
quality from normalization, visual enrichment and source-corrected pipeline results.

Chunking, embedding/retrieval and the RAG agent are secondary supporting studies.
Preserve their comparisons and code, but do not make them the main product story
or the default next milestone unless the user explicitly requests that work.
The user's priority is an evidence-led, visually engaging GitHub presentation.

# Supported dots.mocr execution path

The user explicitly retired the unquantized BF16 CPU approach. Use the retained
Q4_K_M language model and Q8_0 projector on GPU through the CUDA 12.6 native
runtime. Do not recreate CPU downloads, environments, launch profiles or tests
without a new explicit request. Historical CPU raw output/reports and immutable
snapshots are retained as failed evaluation evidence, not an active backend.
The approved WSL 12 GiB memory limit remains configured; no further host-setting
change or restart was requested.

# Evaluation artifact retention

The user explicitly requested preserving these outputs for future evaluation:
the original two-page multimodal PDF, the first 20 NASA financial-report pages,
and the final five-page handwriting/mixed-content selection. Preserve the stopped
20-page and earlier five-page attempts separately; their outputs are partial.

The registry is `evaluation/preserved-runs.json`. Frozen local snapshots live in
`outputs/evaluation_archive/`, with SHA-256 inventories and runtime/code metadata.
Do not delete or overwrite these snapshots unless the user explicitly requests it.
When extraction or normalization changes, preserve a new snapshot separately.

The earlier diary selection completed five pages. Quantized dots.mocr now
also has two preserved runs of the original two-page input, using separate
project/upstream prompts. Use `docs/evaluation/dots-mocr-local-validation.md` for
their quality limitations. Check run status and archive indexes before claiming
any future run is complete.

Model predictions are not verified reference annotations. Keep private source PDFs
and generated archives out of Git and package distributions; preserve source-page
mappings and distinguish a completed subset from a completed full document.

Organize preserved outputs by exact model name/version, e.g.
`outputs/evaluation_archive/PaddleOCR-VL-1.6_output/<case_id>/...`.
Record model identity in the preservation registry and archive index. Keep future
models in separate namespaces so evaluation baselines cannot be mixed.

The quantized GPU batch `01` attempted NASA 20 pages (20 accepted) and the diary
selection five pages (three accepted; source pages 5 and 7 hit the output token
limit). Both are archived under the exact quantized model namespace. Preserve
failures and report the diary as partial; see docs/evaluation/quantized-gpu-batch.md.

# Current handwriting showcase

The user replaced the demo’s historical diary with four photos of their own
mathematics assignment, ordered by handwritten page numbers 1–4. The current
case is `author-handwritten-mathematics-four-pages-01`. Preserve its source JPEGs,
shared PDF, predictions and per-model archives, as well as all earlier diary runs.
See `docs/evaluation/author-handwritten-mathematics.md` and the preservation
registry for results. Selected assistant-reviewed math/text checks and ordered
truth-table cells are not full-document accuracy or independent human labels.

# Current production parser

The production CLI/API and container now default to dots.mocr Q4_K_M language
weights plus Q8_0 projector. Original BF16 dots was explicitly authorized for
Colab GPU comparison; this does not restore the retired CPU path. Its best
selected handwritten-mathematics score is 32/37, with 8.53 GiB peak PyTorch
allocation on T4. Do not call it a universal winner or confuse allocation with
total VRAM/minimum capacity. Historical Paddle IRENA/IMF outputs stay labeled
as Paddle and are not relabeled as results from the new default.
