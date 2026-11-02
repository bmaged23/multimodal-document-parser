# Final source-quality repair

Completed locally on 2026-10-31. The 46 IRENA/IMF pages and original two-page
regression input reused their existing raw OCR predictions. **Zero GPU OCR pages
were rerun.** One bounded Gemma visual batch completed in 149 seconds; subsequent
source checks, corrections and exports use CPU/PDF objects only and make no model
calls. Original runs and raw model responses remain preserved.

## Result and scope

| Selected source assertion | Initial baseline | Final pipeline output |
| --- | --- | --- |
| Table cell text | 277/277 | 277/277 |
| Merged-cell span attributes | 40/40 | 40/40 |
| Sampled column-header roles | 0/32 | 32/32, explicitly marked inferred |
| Printed chart values, series/category matched | 141/144 | 144/144 |
| Chart series units | 38/39 | 39/39 |
| IRENA source visuals represented | 14/15 | 15/15 |
| No invented categories on unlabelled scatter charts | 1/2 | 2/2 |
| Original chart/flowchart/image regression | 9/9 | 9/9 |

These are selected single-assistant development assertions, not full-document
accuracy. Header-role inference is heuristic. **144/144 is the final pipeline
score after native-PDF source correlation, not Gemma's VLM accuracy.** A revised
Gemma image/prompt/review run alone scored 135/144; raw predictions and that
regression are retained. The [five-case raw VLM comparison](five-model-vlm-benchmark.md)
also separates draft and review results from all source corrections.

## Repairs

- Infer initial header bands conservatively from spans, labels and transitions
  to numeric rows. Respect explicit `th`/`thead`; expose `header_inferred` and
  `header_inference` rather than silently presenting guesses as extracted facts.
- Associate all 19 table fragments with their source captions. Link nearby notes
  by source element identity. IMF Table G's ten fragments share a logical table
  ID and ordered continuation indexes, while alternating indicator groups remain
  separate in `column_group`. Table E's two continuations are linked as well.
  No lossy concatenation of unlike columns is performed.
- Recover the missing chapter photograph from large uncovered native PDF image
  bounds. Existing visual/table coverage and dense text coverage suppress
  duplicates and whole-page scan false positives. Small or flat background
  objects and rotated pages are conservatively skipped.
- Use high-resolution PDF-rendered chart crops and a bounded model review.
  Preserve both raw responses. Model review remains fallible and is not a
  verification label.
- Remove null-valued axis-tick categories from unlabelled scatter analysis.
  Retain unknown categories on other plot types. Normalize missing/spelled-out
  units only from an unambiguous explicit chart note, with an adjustment record.
- Detect missing units, duplicate keys, unsupported native text tokens and
  conflicting repeated-chart values. Warnings are visible in HTML.
- Correlate native PDF line paths, legend swatch colors, year labels and printed
  numeric text on the two crowded multi-line charts. Every recovered number is
  copied from a printed text object. Axis calibration checks the association;
  it never generates a value from pixel height. Preserve coordinates, colors,
  labels and method details in per-chart evidence files.
- Distinguish source-extracted chart numbers from generated descriptions in
  JSON/Markdown/HTML. The supported native endpoint extractor produces a disclosed
  source-based endpoint summary, replacing misleading universal-decline prose.
  Raw VLM descriptions remain unchanged in their response files.
- Render recognized footnote superscripts safely, retaining original canonical
  strings. Keep dense tables horizontally scrollable instead of splitting
  country names into unreadable fragments. Arbitrary model HTML/math is not run.

Native endpoint recovery deliberately abstains for raster plots, transformed
forms, nonlinear axes, missing/ambiguous legends or numeric labels, and unsupported
categories. It requires multiple consistently aligned vector lines and a linear
axis. A narrowly permitted single ligature/text-encoding fallback is recorded in
the legend evidence. Other charts retain generated provenance. Native-token
occurrence alone is not proof that a value belongs to a particular series.

## Verification and artifacts

All 48 final report pages passed Chromium navigation and output-tab checks.
All 18 Markdown visual assets loaded (15 IRENA and three original); the IMF
report correctly has zero photo/chart crops. The browser checked 5,081 predicted
table cells and 61 merged cells against JSON. This tests renderer fidelity, not
5,081 source-verified values. Both large reports and the original regression
passed narrow-viewport and JavaScript-error checks.

Final reports:

- `outputs/parser_runs/irena-final-source-checked-03/review.html`
- `outputs/parser_runs/imf-final-source-checked-03/review.html`
- `outputs/parser_runs/original-final-source-checked-03/review.html`

Detailed scored assertions and browser evidence:
`outputs/quality_evaluations/final-source-checked-03/`.
The original frozen baseline and revised raw Gemma run remain separate.

Reproduce source-only postprocessing without OCR or model calls:

```bash
uv run python scripts/finalize_saved_visual_quality.py \
  outputs/parser_runs/irena-final-repair-paddleocr-gemma4-31b-02 \
  --output-directory outputs/parser_runs/irena-source-check-reproduction

uv run python scripts/evaluate_source_review.py \
  outputs/parser_runs/irena-source-check-reproduction \
  evaluation/reference_annotations/irena-mixed-content-20-source-review-v1.json \
  --output outputs/quality_evaluations/reproduced-source-checks/irena.json
```

The processing modules never read expected-answer files; those are opened by the
separate scorer. Checksums bind references, evaluated documents and scoring code.
The source-only postprocessor verifies the input PDF identity, writes a fresh
directory and preserves raw draft/review response bytes.

No RAG chunking has started. A five-model VLM comparison is separately prepared;
four alternatives currently require Ollama billing access and have no quality
scores. No model winner or universal accuracy claim is inferred from that block.

Final local software validation: **97 tests passed**, Ruff formatting/lint and
strict typing passed for 29 source files, and both wheel and source distribution
built successfully. The package excludes local PDFs, outputs, credentials and
model files. Hosted CI and deployment remain unrun.
