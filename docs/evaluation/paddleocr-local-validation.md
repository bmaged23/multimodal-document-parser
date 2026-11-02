# PaddleOCR-VL-1.6: initial local validation

This is an engineering case study on one local two-page synthetic sensor report,
not an OCR benchmark or an estimate of general accuracy. The source document and
its generated artifacts are not distributed in the repository.

## Execution

- Environment: WSL2, Python 3.12.15, NVIDIA GTX 1660 Ti (6144 MiB VRAM).
- Runtime: PaddleOCR 3.6.0, PaddleX 3.6.1, PaddlePaddle GPU 3.2.1.
- Input: two rendered page images, each 1190 × 1684 pixels at 144 DPI.
- Pipeline: complete layout detection and VLM recognition, version `v1.6`.
- Orientation classification and document unwarping: disabled for source alignment.
- Chart recognition: default disabled; chart and figure crops are preserved.

| Observation | Result |
| --- | ---: |
| Successful input pages | 2 / 2 |
| Extracted elements | 34 |
| Table dimensions | 4 rows × 5 columns |
| Extracted table cells | 20 |
| Chart regions | 1 |
| Image regions (including a flowchart) | 2 |
| Initialization, including first model downloads | 227.62 seconds |
| Page 1 recognition pipeline | 89.81 seconds |
| Page 2 recognition pipeline | 42.77 seconds |
| Inference process total | 360.22 seconds |
| Highest sampled total device memory | 5959 MiB |

Memory was sampled approximately once per second using nvidia-smi. It includes
other GPU processes and can miss transient peaks; this is not allocator peak
memory. Initialization timing includes downloads, so it is not warm-start latency.
These figures come from `raw/paddleocr/inference-manifest.json` in the local run.

## Manual inspection

All 20 table cell strings matched the visible source in a manual spot check.
This is a single-table observation; no TEDS or dataset-level score was computed.
The chart, vector flowchart, and raster figure were retained as source crops.
The vertical QC label was recognized, but its place in the reading order needs
review. A section heading on page 2 was classified as ordinary text by the model.
The repeated document title remained a separate heading on each page.

Heading levels are inferred from backend labels/Markdown and have not been
independently scored. Inline mathematical expressions remain in text where the
backend did not produce formula blocks. Table header flags follow backend HTML;
the sample uses `td` for its header row, so header semantics were not inferred.
Chart numbers, image descriptions, and flowchart semantics have not been extracted
into structured enrichment fields; that work belongs to milestone 5.

## Reproduction and inspection

Follow `docs/development/paddleocr-parsing-and-html-review.md` to prepare and parse
an available PDF. Save the source hash, preparation settings, backend lockfile,
raw results, inference manifest, logs, and run summary for each execution.
The same input bytes retain the same document ID across render resolutions.

The reviewed local bundle was regenerated from the actual saved GPU results
without rerunning OCR after a caption-label normalization adjustment. Its
`run-status.json` correctly records `mode=import`; its copied inference manifest
retains the original GPU measurements. The original inference run remains intact.

Software checks use labeled synthetic result fixtures and generated PDFs. They
verify normalization, export, and review interactions, not the model's accuracy.
A varied, distributable annotated dataset, fixed scoring protocol, and repeated
warm/cold performance measurements remain required for the final evaluation phase.

## Practical limits

The run fit on this laptop but used almost all of its VRAM. Larger inputs and
other workloads may exceed available memory. Do not assume that dots.mocr or an
additional vision model will fit concurrently. GPU integration tests must stay
separate from ordinary CPU CI.
