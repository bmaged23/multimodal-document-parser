# Three-model handwritten mathematics comparison

Four photographs of the project author's own assignment replace the historical
diary in the current interactive demo. The original diary predictions, failures,
public v1/v2 demo assets and historical measurements remain preserved.

The pages contain prose, set notation, six drawn Venn diagrams, Cartesian
products, function composition and injectivity/surjectivity arguments, followed
by a handwritten truth table. They are phone photographs with shadows, faint
ink, corrections and crossed-out work. This is a deliberately difficult small
development case, not a standardized handwriting benchmark.

## Input and execution

The handwritten page numbers establish order 1–4. Original JPEGs are retained
locally with SHA-256 hashes. They are embedded in a four-page PDF without image
enhancement or JPEG recompression, then rendered once at 144 DPI. All three
backends receive that same prepared source bundle. Model-specific internal
resizing and region detection still differ; equal source images do not make
this a controlled architecture or precision comparison.

Hardware: **NVIDIA GeForce GTX 1660 Ti, 6,144 MiB VRAM**, WSL2. Models run
sequentially, using the existing isolated uv environments and pinned weights:

- PaddleOCR-VL-1.6, GPU FP16, full parsing pipeline.
- dots.mocr, GPU Q4_K_M language weights and Q8_0 projector, upstream layout prompt,
  6,144 output-token limit, 1,000,000-pixel input cap and 300-second HTTP timeout.
- Unlimited-OCR 2, NF4 language / FP16 vision, experimental compatibility path,
  base view without crops, 4,096 output-token limit and 720-second page-job timeout
  including model loading. Its raw generation is not canonical project JSON.

The retained profiles have different token limits and time budgets. PaddleOCR
has no imposed wall-clock page cap in this run; dots and Unlimited use the bounded
limits above. A timeout is an operational failure, not proof of what a model
could transcribe with unlimited time. These are practical profile comparisons,
not a controlled architecture or equal-budget benchmark.

No cloud visual model, source correction or answer repair is applied. The source
assignment's mathematical mistakes remain part of the transcription target.

[Input selection and hashes](../../evaluation/test_cases/author-handwritten-mathematics-four-pages.json)
· [Retained model profiles](../parsing-evaluation/README.md#models-and-execution-scope).

## Reference and scoring mechanism

Before reading predictions, the assistant visually selected **37 readable
assertions: 18 prose spans and 19 mathematical expressions**. The handwritten
truth table adds **20 cells in four rows and five columns**. These are selected
assistant-reviewed references, not independent human labels or a full-page
transcription. The author's independent verification would improve confidence.

The scorer checks each assertion on its own page. Case, whitespace, ordinary
prose punctuation, TeX formatting and explicitly listed notation variants are
normalized. Mathematical negation, exponents, set boundaries and operators stay
significant. An unfinished repetitive set expression cannot satisfy a complete
set reference. Ambiguous symbols, the cropped top line on page 1, wholly crossed-out
answers and uncertain shading boundaries are excluded.

The truth table must be recovered as an HTML or Markdown table with the expected
ordered headers. Cells are compared at their original row and column positions;
finding the same letters elsewhere in prose does not pass. A zero can mean that
the required structure or headers were not recovered; it does not imply every
individual character was misread. Missing outputs score
zero on their requested checks. Extra assertions or hallucinations do not reduce
this presence-based score, so it must be read alongside the qualitative failures.

Metrics are reported separately; they are not combined into a single accuracy
percentage. Full-page CER/WER, diagram-semantic accuracy and hallucination
precision are **not measured**. No Ragas or LLM-as-judge score is used here.

[Frozen selected references](../../evaluation/reference_annotations/author-handwritten-mathematics-selected-v1.json)
· [Scoring implementation](../../scripts/parser_evaluation/evaluate_handwritten_math_comparison.py).

## Observed results

**Finding: quantized dots.mocr was substantially better at handwritten text and
mathematical notation on this assignment.** It recovered 29/37 selected checks,
versus PaddleOCR's 13/37 and Unlimited-OCR's 11/37, and completed all four pages.
We prefer this tested dots profile for similar handwritten mathematics PDFs,
while reviewing formulas and proofs against the source. The evidence is limited
to these four pages and the retained configurations; it does not establish a
universal handwriting ranking or the performance of unquantized dots.mocr.

| Tested profile | Completed pages /4 | Text checks /18 | Math checks /19 | Ordered table cells /20 | Median page time |
| --- | ---: | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.6 | 4/4 | 8/18 | 5/19 | 20/20 | 420.4 s |
| dots.mocr Q4_K_M + Q8_0 | 4/4 | 17/18 | 12/19 | 20/20 | 23.5 s |
| Unlimited-OCR 2 NF4/FP16 | 2/4 | 9/18 | 2/19 | 20/20 | 139.2 s |

The median uses available runtime page/generation measurements, excluding model
loading. Unlimited's median covers **only its two successful pages**; its failed
page jobs each consumed approximately 720 seconds including loading. These
unequal scopes make a direct speedup claim inappropriate. Per-page inference,
loading and attempt durations are retained in the JSON results.

**dots.mocr recovered the most selected text/math in this case**, but it still
misread a function name as `9`, invented/changed Cartesian-product entries and
rewrote parts of the surjectivity argument. It also added a negation to the
proposition definition on page 4. Its high selected-span score does
not establish that the full mathematical proof is correct.

PaddleOCR's first page contains a long repetitive set fragment (the most frequent
16-token sequence occurs 492 times). It took 1,263.7 seconds. Page 3 omitted much
of the lower function work; nevertheless page 4's structured truth table was
recovered correctly. Saving four outputs is not four accurate transcriptions.

Unlimited completed pages 2 and 4; pages **1 and 3 timed out**. It changed several
logical/set operators and some words even in successful pages, but preserved
six predicted Venn image regions on page 2, compared with three Paddle image
regions and two dots image regions. These counts are diagnostics, not scored
box recall or diagram-semantic accuracy. All six drawings remain visible in the
source. Canonical exports may also contain native full-photo recovery; that
pipeline fallback is not credited as model recognition or scored transcription.

Returned TeX is preserved; the reader does not fully typeset every mathematical
expression or repair its notation. Inspect the raw and Markdown views when
checking formula fidelity.

Reading order was inspected qualitatively through the source/output reader, not
against independent reading-order labels. Dense missing/repetitive regions are
visible; no reading-order accuracy percentage is claimed. No model consistently
transcribed every page correctly, and none's diagrams were semantically verified.

[Per-assertion, per-cell, timing and repetition results](../../evaluation/results/author-handwritten-mathematics-comparison-v1.json)
include source/reference/evaluator hashes and hashes of the scored page text.
The evidence dialog in the [interactive demo](https://bmaged23.github.io/multimodal-document-parser/?v=3.0&case=handwriting&run=math-paddle&page=4)
shows the same selected checks. All original predictions, failures and logs are
preserved in model-specific local snapshots. Earlier diary and RAG studies
remain frozen; this case does not silently replace their evaluation inputs.

## Reproduce locally

The source photos and model weights are not shipped in a fresh clone. Supply the
four original photos locally, install the documented GPU runtimes and use the
existing project environment:

```bash
uv sync --locked --group dev
uv run --no-sync python scripts/parser_evaluation/run_handwritten_math_comparison.py \
  --downloads /path/to/photos --stage all
uv run --no-sync python scripts/parser_evaluation/evaluate_handwritten_math_comparison.py
```

The preparation script records the original names and orders pages by the
assignment's visible page numbers. Existing outputs are never overwritten;
use a new case/run identifier for another inference attempt. Individual
`--stage paddle`, `--stage dots` and `--stage unlimited` runs are available.
Predictions, raw replies, logs, page-level status and model-named checksum
archives remain local; sanitized metrics and the public saved-output demo are
published separately.
