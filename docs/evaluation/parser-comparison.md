# Parser comparison

Use the same preparation bundle for both parsers. The comparison rejects different
source hashes, page geometry, DPI, and rendering settings. It accepts completed
subsets of interrupted runs only when every selected page has a saved inference
checkpoint. Raw data and older evaluation snapshots are never overwritten.

```bash
uv run document-parser compare \
  outputs/parser_runs/paddleocr-run \
  outputs/parser_runs/dots-mocr-run \
  --output-directory outputs/parser_comparisons/example
```

`comparison.html` embeds both existing review reports and links to
`comparison.json`. Serve the parent output directory to keep report/assets links
available. The JSON contains model identities, raw inference metadata, per-page
latencies, text/element/table-cell counts, and whitespace-normalized word-sequence
agreement. Use `--pages 1 2` to choose a completed subset.

Agreement is not accuracy: two models can agree on an error. More output and more
table cells do not establish better extraction. Parser preprocessing, quantization,
initialization, and GPU sampling scope must accompany latency/memory comparisons.
PaddleOCR's measurements may include initialization; the dots.mocr HTTP client
excludes server startup. The comparison preserves these distinctions rather than
presenting a misleading speedup. Device memory sampling is not model allocator peak.

## Reference transcription

Optional `--reference reference.json` calculates character and word edit-error
rates against manually verified page transcriptions. The reference file is tied
to the source PDF hash and must identify how it was annotated:

```json
{
  "document_id": "sha256:<64-character source hash>",
  "annotation_provenance": "Transcribed from source and checked by a named reviewer",
  "page_text": {"1": "Complete page transcription in reading order"}
}
```

The comparison includes headings, formulas, headers/footers, and table cell text
in row order, normalizing whitespace only. References must follow that convention.
Scores report edit counts and aggregate rates weighted by reference length;
insertion-heavy predictions can exceed 100% error. No score is created for
unannotated pages. These text metrics do not measure table spans, layout, image
quality, or chart values. Those need task-specific annotations in milestone 8.

The synthetic test fixtures exercise these calculations and failure paths. They
are not real model benchmark results or reference annotations for public datasets.

## Source-bound table checks

A reference may contain `table_cells` instead of (or alongside) `page_text`:

```json
{
  "document_id": "sha256:<64-character source hash>",
  "annotation_provenance": "Describe who checked the source and how; disclose automated references",
  "table_cells": [
    {"page_number": 1, "table_index": 0, "row": 1, "column": 3, "text": "18.4",
     "row_span": 1, "column_span": 1}
  ]
}
```

Table indexes, rows, and columns are zero-based; pages are one-based. Checks use
the table's cell origins and normalize whitespace only. Optional span expectations
check geometry for those cells. Missing tables/cells and wrong values fail checks.
Report passed/checked counts as selected assertions, not overall table accuracy.
References may be source-derived assistant spot checks; identify this explicitly
and do not present them as independently verified human ground truth. Keep private
references under the ignored `outputs/parser_comparisons/` directory.
