# Gemma selection and evaluation checkpoint

Subsequent status: the user approved proceeding after this checkpoint. See
[adaptive RAG development](adaptive-rag-development.md) for the implemented
chunking, GPU retrieval and separate service APIs. The observations below record
the earlier evaluation boundary.

Completed on 2026-10-31 using saved results. The selected visual model is
`gemma4:31b` through Ollama Cloud. The second API key is configured privately.
This is the user's practical selection after confirming access, not a model
quality win. Twelve alternative hosted chat/vision models returned payment
errors; their absent predictions are not accuracy failures.

## Rechecked final pipeline output

Fresh source-only scoring of the 20-page IRENA and 26-page IMF final runs matched
the preserved scores and evaluated document hashes:

| Selected check | Passed / checked |
| --- | --- |
| Table cell text | 277/277 |
| Merged-cell span attributes | 40/40 |
| Inferred header roles | 32/32 |
| Printed chart values with series/category association | 144/144 |
| Chart series units | 39/39 |

These are selected, non-blind, single-assistant development checks. The final
chart scores include native-PDF corrections and are not raw VLM accuracy.
Header semantics remain explicitly inferred. The original two-page regression's
previous 9/9 checks and all raw responses remain preserved; that regression was
not rerun at this checkpoint.

## Raw visual model results

The fixed five-visual sample contains a crowded multi-series line chart, grouped
bar chart, unlabelled scatter plot, equipment photograph and landscape photograph.
Saved raw Gemma responses yield 23/32 printed chart values in the draft and
24/32 after model review. The grouped bar chart passes 18/18; the crowded line
chart improves from 5/14 to 6/14. The scatter response abstains from inventing
numeric point values. Both photos cover 4/4 selected visible features.

Known unsupported claims remain visible in the manual review: motion direction
in the equipment photo, and an incorrect all-technologies-decline statement for
the line chart. Source-based postprocessing repairs the supported native-vector
chart cases, but cannot establish universal correctness for other PDFs.

## Decision and next boundary

Keep Gemma as the cloud default and preserve generated versus source-extracted
provenance, raw draft/review responses, and quality warnings. No further OCR or
cloud generations were needed for this checkpoint. No new local model inference
was started. Cross-model quality ranking, held-out validation and independent
factuality review remain incomplete; the access investigation is documented in
the [VLM comparison](five-model-vlm-benchmark.md).

The next milestone is structure-aware RAG preparation: source-linked JSONL
chunks retaining headings, page regions, table continuation identities, captions,
and visual provenance. **Chunking has not started.** This checkpoint provides the
completion notification requested before that work.

Machine-readable decision, evidence hashes and score scope:
[`evaluation/results/gemma-selection-checkpoint-v1.json`](../../evaluation/results/gemma-selection-checkpoint-v1.json).
Fresh local scoring evidence:
`outputs/quality_evaluations/gemma-selection-checkpoint-01/`.
See [final repairs](final-source-quality-repair.md) for the existing HTML reports,
reproduction commands, browser checks and software validation.
