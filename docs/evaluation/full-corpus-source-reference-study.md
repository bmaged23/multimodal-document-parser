# Source-reference study of the complete retained test corpus

This study covers every page in the four retained test PDFs: NASA 20, handwritten
mathematics 4, IRENA 20 and IMF 26. These selections total **70 pages**, evaluated
against four saved OCR profiles for **280 page/profile slots**. It does not cover
the unselected remainder of the original NASA or IRENA publications.

The references contain **750 annotations, 29 table grids, 5,246 cells, 48 visual
regions and 144 printed chart values**. Every page has reference content. Explicitly
unresolved handwriting and deleted text are excluded from text scoring, with an
exclusion ledger. Chart and diagram labels have separate reference blocks and are
outside the prose error-rate denominator.

## Reference provenance

The initial eight-page study was preserved. Its printed paragraphs used source
text extraction within visually checked regions; tables and handwriting were
drafted against the visible source. The remaining 62 pages were transcribed by
the assistant directly from rendered page images. PDFium rendered the new pages;
it did not extract their text. **PyMuPDF was neither installed nor used.** OCR
predictions and Gemma descriptions did not supply reference answers.

The user inspected examples and endorsed the approach. This is still an
assistant-created, retrospective reference set. No independent human annotation,
blind review or individual user approvals are claimed. All items remain editable
drafts in the reference viewer. Scores against the frozen copy are development
measurements, subject to reference corrections.

## Saved evaluation

The public-safe [metric summary](../../evaluation/results/manual-reference-full-corpus-v1.json)
contains aggregate and per-document results, denominators, limitations and hashes.
Private source text, complete references and detailed traces stay under `outputs/`.

| Development measurement | Paddle | dots Q4/Q8 | dots BF16 | Unlimited |
| --- | ---: | ---: | ---: | ---: |
| Scoped region character error rate ↓ | 3.62% | 0.67% | 0.72% | 3.20% |
| Strict grid cell-text match ↑ | 83.55% | 82.39% | 79.87% | 85.09% |
| Label-linked row/value match ↑ | 86.58% | 72.05% | 80.96% | 55.62% |
| Table cell-content multiset F1 ↑ | 91.61% | 90.36% | 90.50% | 87.53% |
| Visual region recall at IoU ≥ 0.50 ↑ | 81.25% | 85.42% | 91.67% | 85.42% |
| Pages with native layout output | 70/70 | 70/70 | 70/70 | 68/70 |

These measurements test different properties and do not establish one combined
winner. In particular, Unlimited's strict cell-origin score coexists with weaker
label-linked row results. The dots profiles recover printed text well in this
study, and BF16 has the highest visual-region recall. Scoped handwriting text
results differ from the older 37-assertion study because the reference units,
matching rules and excluded ambiguities differ; do not merge their scores.

## What each metric means

- **Region CER/WER:** compare each resolved reference block with the best matching
  contiguous span inside overlapping native OCR blocks. This tolerates different
  paragraph segmentation. It measures scoped content recovery and does not penalize
  extra text outside the matched span. It must not be converted into a claim of
  full-page transcription accuracy. Page-order edit diagnostics are also saved.
- **Strict cells, spans and header flags:** compare exact source-grid positions.
  Alternative valid representations can score poorly. Header/body flag accuracy
  includes body cells, so the denominator is important.
- **Label-linked rows:** match a unique row key and its ordered values. Header
  offsets are tolerated; an objective ID joined to its description is supported.
  Nonunique reference keys, such as repeated truth-table T/F labels, are excluded
  from this metric. The strict grid still covers those cells.
- **Cell-content F1:** compares nonempty cell-text multisets. It does not check
  relationships between countries and values.
- **Visual recall and crop overlap:** use one-to-one geometry matches. Institutional
  logos and decoration are outside scope; false-positive precision is not claimed.
- **Headings and reading order:** detect native title/heading regions and compare
  order among geometrically matched text pairs. Heading levels and parent links
  are not scored when native profiles do not emit those fields.
- **Chart numeric-token recovery:** checks printed numbers in native text within
  chart regions. It is not a series/category association score. The profiles
  retain these figures as visual regions, so this OCR-only diagnostic is zero in
  the saved runs; enrichment quality must be evaluated separately.

Normalization equates whitespace, quote/dash variants, solid-dot glyph variants
and spaced ellipses while preserving the distinction between dots and stars and
between positive and negative numbers. Mathematical LaTeX equivalence is not
fully implemented, so these scores do not claim formula semantic accuracy.

## Reproduce locally

The retained private corpus, frozen references and native evidence packages are
required. No model inference, cloud API or OCR rerun is needed.

```bash
uv run python scripts/parser_evaluation/evaluate_manual_reference_corpus.py \
  --reference outputs/reference_review/full-corpus-source-review-06/dataset.json \
  --output-directory outputs/quality_evaluations/new-manual-reference-evaluation
uv run python scripts/visualization/build_manual_reference_report.py \
  outputs/quality_evaluations/new-manual-reference-evaluation
```

Each fresh evaluation saves `metrics.json`, `page-metrics.json`,
`failure-traces.json`, `excluded-references.json`, `frozen-reference.json` and
`frozen-scorer.py`. Source-image and native-output hashes are checked. The HTML
report adds document filters, page links and a metric-explanation column.
Earlier preliminary scorer runs are preserved separately.

Metric arithmetic was tested against independent dynamic-programming and
exhaustive-assignment implementations. Browser checks covered all 70 reference
pages, all report document filters and the 280 page/profile rows. Neither raw
predictions nor the deployed parser configuration were changed by scoring.
