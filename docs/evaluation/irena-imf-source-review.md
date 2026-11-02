# IRENA and IMF source review

Evaluation completed locally on 2026-10-31. PaddleOCR transcribed the reviewed
table cells well and preserved the checked merged cells. The outputs still have
missing semantic table headers, one missed photograph, and several chart-data
errors. These predictions are a useful development baseline, not verified input
for unattended numerical answers.

The user requested a checkpoint after evaluation and before structure-aware RAG.
This evaluation is complete within the scope below. RAG chunking has not started.

## Inputs and model configuration

| Case | Evaluated pages | Source mapping | Output inventory |
| --- | --- | --- | --- |
| IRENA, Renewable Power Generation Costs in 2024 | 20 | Original physical PDF pages 13–32 | 2 tables / 106 cells, 10 charts, 4 extracted photographs |
| IMF, WEO October 2025 Statistical Appendix | 26 | Complete appendix; printed pages 97–122 | 17 table fragments / 4,955 cells; no chart/photo regions |

Sources: [IRENA PDF](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jul/IRENA_TEC_RPGC_in_2024_2025.pdf)
and [IMF PDF](https://www.imf.org/-/media/files/publications/weo/2025/october/english/statsappendix.pdf).
Local selections pin SHA-256 identities and original page mappings in
`evaluation/test_cases/irena-renewable-power-costs-20.json` and
`evaluation/test_cases/imf-weo-october-2025-complex-tables-26.json`.
Publisher documents and third-party photographs remain local and out of Git.
The repository references transcribe selected factual data, with publisher attribution.

OCR: PaddleOCR-VL-1.6 through PaddleOCR 3.6.0 / PaddlePaddle GPU 3.2.1, FP16,
one model, one page at a time, 144 DPI; orientation/unwarping disabled.
Hardware: GTX 1660 Ti, 6,144 MiB VRAM.
VLM: `gemma4:31b` through Ollama Cloud, catalog digest `221b330d11a8`, temperature
0, output budget 2,048 tokens, source crop plus page context. The cloud deployment
revision is unpinned. IRENA enriched 14/14 detected visuals. IMF had zero visual
candidates, so its enrichment stage made no visual inference calls.

## Annotation and measurement process

1. Validate canonical documents, page checkpoints, source hashes and local assets.
2. Overview all 46 rendered source pages. Inspect enlarged source pages for the
   specific checks, and transcribe references from those sources.
3. Score table text, spans and semantic header flags separately. Score chart
   series/category/value identity, units, and abstention separately.
4. Check selected reading-order anchors. Exercise all 46 pages in Chromium,
   inspect JSON-to-HTML table fidelity, load images, and test narrow viewports.
5. Preserve predictions unchanged; save annotations, scoring code, results,
   browser evidence, checksums and the failure catalog.

This is a **single-assistant, non-blind development evaluation**. The model
inventory was seen before annotation; values were transcribed from rendered
source pages, not copied from predictions. There is no second annotator,
independent human adjudication, inter-annotator agreement, random sample,
confidence interval or held-out test set. Do not describe these results as an
independent benchmark or general model accuracy.

References cover both complete IRENA tables (106 cells), 171 selected IMF cells
on pages 7, 13, 14 and 21, all 144 printed numeric chart labels across eight
IRENA charts, and abstention on two unlabelled scatter plots. IMF selection
includes grouped headers, first/last rows, wrapped country names, blanks,
fiscal-year strings and missing-value markers. Only **277 of 5,061 output table
cells** were source checked. Thumbnail overview is not a transcription audit.

Cell text matching normalizes whitespace only and preserves case, signs and
decimal formatting. Chart series/category labels ignore case and whitespace
only; numerical tolerance is 1e-9, and values must claim `printed_label` provenance.
Value checks require the correct series and category, not mere occurrence of a
number elsewhere. Unit correctness has its own denominator. Reading-order
anchors must be unambiguous; repeated short phrases are annotation errors.

## Measured results

| Property | IRENA | IMF | Meaning |
| --- | --- | --- | --- |
| Completed parser pages | 20/20 | 26/26 | Operational completeness |
| Table count by source page | 20/20 | 26/26 | Page-level counts, including pages with zero tables; not cell accuracy or box IoU |
| Selected cell text at exact row/column | 106/106 | 171/171 | Source transcription checks only |
| Selected row/column span attributes | 14/14 | 26/26 | Both attributes on 20 sampled merged cells |
| Selected semantic column-header flags | **0/19** | **0/13** | All sampled headers have `is_header=false` |
| Reading-order anchor pairs | 8/8 | 8/8 | Selected adjacent anchors on IRENA pages 3/7 and IMF page 1 |
| Visual count by source page | 19/20 | 26/26 | One IRENA chapter-divider photograph missed |
| Expected visual kind, including missed image | 14/15 | N/A | All 14 detected visuals classified correctly; one image absent |
| Printed chart values by series/category | **141/144 (97.9%)** | N/A | Two swapped values and one missing endpoint |
| Generated non-null points supported by source labels | 141/143 (98.6%) | N/A | Same errors viewed from prediction precision; not an additional independent sample |
| Chart series units | **38/39** | N/A | Missing GW on regional capacity chart |
| Numeric abstention on unlabelled charts | 2/2 | N/A | No fabricated numerical point values |
| No invented categories on unlabelled charts | **1/2** | N/A | One chart emits 23 axis ticks as null-valued categories |

There is no combined score: these properties have different units and overlapping
evidence. The 16 reading-order checks are not a full layout benchmark. Full text
CER/WER, full-table tree edit metrics, bounding-box IoU, image-caption factuality,
hierarchy accuracy and multi-page table assembly were not quantitatively scored.
No fresh dots.mocr or Unlimited-OCR 2 run was performed on these inputs; these
results do not rank the three engines.

## Failure catalog

All IRENA sample page numbers below map to original physical PDF page `sample+12`.

| ID | Source location | Observation | Implication |
| --- | --- | --- | --- |
| Q001 | Both PDFs, sampled table headers | 0/32 sampled semantic header checks pass; inventory shows zero header flags across all 19 tables | Cell spans survive, but downstream code cannot treat `is_header` as a reliable header map |
| Q002 | IRENA sample 4 / original 16, Figure S1, `element_00036` | CSP 2024 should be 0.092, predicted 0.087; bioenergy should be 0.087, predicted 0.092 | Plausible numbers attached to the wrong series; `printed_label` is a model claim, not proof |
| Q003 | Same chart | Geothermal 2010 is visibly labelled 0.055 but omitted | Enrichment can silently omit legible values |
| Q004 | IRENA sample 9 / original 21, Figure S6, `element_00079` | 23 x-axis ticks emitted as null-valued point categories | Correct numeric abstention still leaves misleading chart structure |
| Q005 | IRENA sample 15 / original 27, Figure 1.1, `element_00133` | All 18 bar values match, but series unit is null despite the source note defining GW | Units must accompany numbers; this is scored separately from value correctness |
| Q006 | IRENA sample 14 / original 26 | Transmission-tower chapter photograph absent from raw parser blocks and canonical visual elements; only title/credit extracted | VLM never receives missed regions; 14/14 enrichment does not mean all source visuals were captured |
| Q007 | IMF Table G, sample pages 13–22 / printed 109–118 | Ten page fragments stay separate; captions are text elements and table `caption` fields are null | No implemented logical table identifier, continuation link or column-group merge across pages. Alternating page layouts contain different indicator groups; naive concatenation would be wrong |
| Q008 | IMF sample 13 HTML table header | Footnote superscripts appear as literal LaTeX delimiters, and narrow columns wrap country names heavily | Render is inspectable but typographic fidelity is limited; do not confuse this with missing table values |

The four detected photographs were visually reviewed for broad visible content:
turbine/solar panels, waterfront solar panels with skyline, turbine maintenance,
and a dam. This is a qualitative review, not a scored claim that all details or
landmark identifications in their captions are independently verified.

## HTML and export validation

Both reports passed actual Chromium checks for every page's synchronized
navigation, current-page filter, Content/Markdown preview/Markdown source/JSON
views and downloadable PDF/JSON/Markdown. All 14 IRENA Markdown images loaded;
IMF correctly has zero chart/photo crops. No JavaScript errors occurred and both
390-pixel viewport checks passed.

All 5,061 exported table cell tags were compared with canonical JSON, including
61 cells with row/column spans. This checks **renderer fidelity to predictions**;
it does not source-verify 5,061 cell values. Source crops/page assets exist.
Nonvisual OCR elements are byte-for-byte equivalent after JSON loading before
and after enrichment. Dense table readability and literal footnote syntax remain
documented limitations despite the functional browser checks passing.

## Runtime and recovery

| Measurement | IRENA | IMF |
| --- | --- | --- |
| Recorded cumulative inference time | 1,407 s (23.4 min) | 8,408 s (140.1 min) |
| Mean completed-page time | 68.2 s | 313.0 s |
| Median completed-page time | 62.5 s | 185.3 s |
| Slowest completed page | 213.6 s | 1,710.1 s (28.5 min), page 19 |
| Peak sampled total device memory | 5,959 MiB | 5,959 MiB |

IMF's dense Table G pages 13–22 had a median completed-page time of 497.8 s
(8.3 min). The initial attempt failed with CUDA error 999 while processing page
16; restart reused 15 checkpoints. The user later paused at 19 pages, then a
second restart reused those checkpoints and completed all 26. Failed and final
snapshots remain separate.

These are logged local observations, not hardware-normalized speed benchmarks.
Cumulative inference manifests preserve checkpointed attempt timings; they are
not end-to-end wall time and omit some failed/uncheckpointed work and user pause
intervals. Page times describe completed attempts, not all retries. GPU memory
is whole-device usage sampled once per second, not an exact allocator peak.

## Reproduce without model inference

The existing uv environment is sufficient. Saved model outputs are local and
ignored by Git; a fresh clone must reproduce or obtain the corresponding runs.
Use fresh destination paths: the scorer refuses to overwrite a results file.

```bash
uv run python scripts/parser_evaluation/evaluate_source_review.py \
  outputs/parser_runs/irena-mixed-content-20-paddleocr-gemma4-31b-01 \
  evaluation/reference_annotations/irena-mixed-content-20-source-review-v1.json \
  --output outputs/quality_evaluations/reproduced/irena-results.json

uv run python scripts/parser_evaluation/evaluate_source_review.py \
  outputs/parser_runs/imf-complex-tables-26-paddleocr-gemma4-31b-01 \
  evaluation/reference_annotations/imf-complex-tables-26-source-review-v1.json \
  --output outputs/quality_evaluations/reproduced/imf-results.json

uv run --group browser python scripts/visualization/verify_saved_review_reports.py \
  outputs/parser_runs/irena-mixed-content-20-paddleocr-gemma4-31b-01 \
  outputs/parser_runs/imf-complex-tables-26-paddleocr-gemma4-31b-01 \
  --output-directory outputs/quality_evaluations/reproduced/browser
```

Detailed local results: `outputs/quality_evaluations/irena-imf-source-review-02/`.
Repository-safe measured summaries: `evaluation/results/irena-imf-source-review-v1.json`.
References pin source identity; results pin reference, evaluator and canonical
JSON hashes. Evaluation does not call a model or alter the parser/enrichment
outputs. A draft repeated-word reading-order anchor was corrected before the
final score; the first draft results remain local and are not the final report.

The new evaluator has 10 synthetic regression tests covering swapped chart
series, absent units, missing cells, signs, semantic headers versus spans,
null-valued fake categories, ambiguous anchors, duplicate predictions and
invalid source identity/scope. The complete local suite passed **76 tests**;
Ruff formatting/lint and strict source type checks passed. Synthetic test
success measures software behavior, not model accuracy. Hosted CI has not run.

## Checkpoint before RAG

No chunking code or index was created. Before implementing RAG, review the
semantic-header failure, preserve table/footnote/caption associations, and define
how unverified chart data and missed visuals are flagged. Fixes should create
new output versions and be rescored against these references; retain this
baseline. A later unbiased model comparison needs additional independent
annotations and identical page selections for every engine.
