# dots.mocr local GPU validation

Milestone 4 is complete locally for integration and a two-page development
comparison. This is a small compatibility/quality spot check, not a general
benchmark or held-out evaluation. A subsequent [GPU batch](quantized-gpu-batch.md)
accepted all 20 NASA pages and three of five diary pages; two diary pages reached
the output token limit. Their PaddleOCR outputs and interrupted attempts remain preserved.

## Configuration and provenance

- GPU: NVIDIA GTX 1660 Ti, 6,144 MiB; driver 560.76 under WSL2.
- Quantized language model: Q4_K_M; vision projector: Q8_0.
- Conversion: `prithivMLmods/dots.mocr-GGUF`, revision
  `5cd9f01392968881af86dfde7bc60b9cb412f87a`.
- The converter identifies dots.mocr as its base but does not pin its original
  source revision. Equivalence to a specific BF16 release is unverified.
- Runtime: llama.cpp b11960, built locally with CUDA 12.6.85 for sm75.
  Source/bootstrap hashes, exact native package URLs/checksums, and build options
  are pinned in `backends/dots_mocr_runtime/`.
- One server request at a time; six CPU threads; 8,192-token context; flash
  attention, PDL, and CUDA graphs disabled.
- The same original two-page PDF/preparation bundle was used for both parsers
  (source SHA-256 `be2ebf36f35c4b66f958ef7e7c2b273ad704b9ff3e6748df1942435ff3257cde`,
  rendered at 144 DPI). dots.mocr receives an explicitly preserved 840 × 1,176
  image per page under a one-million-pixel budget. Model preprocessing differs
  from PaddleOCR and must not be treated as an isolated quantization experiment.
- Greedy decoding, top-p 1, 4,096 output-token cap, 300-second socket timeout.

The CUDA 12.8 prebuilt binary failed warmup with unsupported PTX on this driver.
Disabling PDL alone did not fix it. The isolated CUDA 12.6/sm75 source build ran
successfully without changing the system driver. Setup/build tools remain in the
project cache; the installable Python package uses the existing uv environment.

## Measurements

| Run | Page 1 inference | Page 2 inference | Sum of page times | Sampled device memory |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.6 preserved baseline | 89.81 s | 42.77 s | 132.58 s | 5,959 MiB |
| dots.mocr Q4 + Q8 projector, project prompt | 29.26 s | 24.16 s | 53.42 s | 5,901 MiB |
| dots.mocr Q4 + Q8 projector, official prompt | 33.77 s | 24.09 s | 57.86 s | 5,901 MiB |

These page times exclude model initialization/downloads and exports; backend
work and image preprocessing differ. The HTTP client also excludes server
startup. Memory is GPU 0 total-device usage sampled once per second, including
other allocations, not a model allocator peak. PaddleOCR's original measurement
window also included initialization; dots.mocr sampling starts after server
startup. These observations cannot establish a universal speed or memory advantage.
The device still approaches its 6 GiB limit despite quantized weights.

Both dots.mocr runs completed normally, preserving full responses, usage, sent
images, normalized JSON, Markdown, source crops, HTML, logs, and per-page timing.
The server was stopped after each validation to release GPU resources.

## Source-derived quality checks

The coding assistant transcribed the 20 sensor-table cells from source page 1,
visually checked each value against the rendered page, and compared cell text,
origin row/column, and row/column spans. Whitespace is normalized. Neither parser
received these checks as inference input. These are source-derived assistant
spot checks, not independently verified human annotations.

All three runs passed **20/20** selected cell checks. This establishes successful
extraction of this simple table only; it does not establish overall OCR, complex
table, layout, or chart accuracy. No real CER/WER score was computed: the optional
reference-text scoring code is tested using synthetic fixtures, and a complete
verified page transcription has not been supplied.

## Visual review and limitations

Both prompts preserve readable text and all table values on this development
sample. The rotated label on page 2 is recognized. The official prompt produces
more explicit header/footer categories and is the documented default. Its
upstream revision is `23f3e5612fb8066d4034d5ecfc8f33a9243533eb`; the exact prompt,
SHA-256, source metadata, and MIT attribution are packaged with the adapter.
The earlier project-prompt results remain separate for inspection.

Bounding boxes are a material limitation. In both runs, the chart crop cuts
through the bars and omits the bottom axis. The official-prompt flowchart crop
misses its right branch; the earlier flowchart box extends beyond the supplied
image. The official run also has one clipped box. Source highlights consequently
require visual review. These predictions are retained, not silently repaired.
PaddleOCR has its own layout/reading-order limitations; consult its validation
report before choosing a backend.

Dots.mocr's `Picture` category maps to canonical `image`; a chart is not classified
as a chart automatically. Numeric chart reconstruction, image descriptions,
complete mathematical rendering, and corrections to model heading levels are
not established by this milestone. Before milestone 5 relies on a crop for
visual understanding, it must consider full-page context and incomplete crops.

## Reproduce and inspect

See [runtime setup](../../backends/dots_mocr_runtime/README.md) and the
[comparison guide](parser-comparison.md). Actual local runs:

- `outputs/parser_runs/dots-mocr-q4-k-m-local-validation-01/`
- `outputs/parser_runs/dots-mocr-q4-k-m-local-validation-02-upstream-prompt/`
- `outputs/parser_comparisons/original-two-pages-paddleocr-vs-dots-mocr-q4-upstream-prompt/`

The model namespace is
`outputs/evaluation_archive/dots.mocr-Q4_K_M-mmproj-Q8_0_output/`.
Its immutable snapshots include source-bound checks and comparison JSON. The
private input, references, and outputs are local and ignored by Git. Archived
source code is explicitly the code at snapshot time; it is not a claim that all
historical execution revisions were pinned.

## Quantization and interpretation

The demo uses Q4_K_M (4-bit) language weights and a Q8_0 (8-bit) vision
projector. Full precision could change quality, but no matched GPU experiment
is available to quantify that effect or establish a different winner. Earlier
CPU experiments used different execution conditions and are not a controlled
quantization comparison. Report these results as configuration-specific.
