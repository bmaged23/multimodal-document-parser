# Evaluation document corpus

The initial public-source corpus contains four original PDFs and 410 PDF pages.
The files were downloaded and inspected locally. No inference scores are claimed
for this corpus yet. These documents supplement the original private two-page test.

| Source | PDF pages | Cases covered | Initial development pages |
| --- | ---: | --- | --- |
| NASA FY2024 Agency Financial Report | 127 | Financial tables, stacked/donut charts, organizational diagram, photographs, Spanish appendix | 11, 30, 50, 68, 118 |
| The AI Scientist, arXiv 2408.06292v3 | 186 | Scientific text, diagrams, multi-panel plots, equations, tables, code-like text, two-column appendices | 3, 7, 106, 154 |
| NASA ESTO 2024 Annual Report | 21 | Wide two-page spreads, infographics, multi-column articles, text on dark backgrounds, photographs | 3, 8, 10 |
| Lucinda Davis Diary, 1874 | 76 | Real historical cursive, faded handwriting, curved book spreads, mixed print/handwriting, scanned calendar | 5, 12, 35, 74 |

Page numbers above are **1-based PDF pages**, not printed page labels. The ESTO
PDF has wide spreads containing two printed pages. All 76 pages of the diary lack
a PDF text layer. Diary pages 12, 35, and 74 were visually checked as handwriting;
page 5 contains a printed calendar/table and is a distinct mixed-content case.
The diary includes covers and blank/printed material, so its 76 PDF pages must
not all be counted as handwritten-text pages.

The scientific paper's appendices include AI-generated research papers, explicitly
marked in the source. They are useful layout/extraction cases; their scientific
claims are not independently verified. This is not an independently sourced
collection of 186 research articles.

## Files and provenance

Local originals: `data/input_documents/evaluation_corpus/` (ignored by Git).
The versioned `evaluation/corpus-manifest.json` records stable IDs, descriptive
filenames, source/download URLs, SHA-256 hashes, byte/page counts, usage notes,
text-layer presence, and the initial selected pages.

Download or verify the exact files from the repository root:

```bash
uv run python scripts/document_processing/download_evaluation_documents.py
uv run python scripts/document_processing/download_evaluation_documents.py --document-id lucinda_davis_diary_1874
```

The downloader checks pinned hashes, rejects non-PDF responses, and writes to a
temporary file before publishing a completed download. Existing files with a
mismatched hash are not replaced. Source changes must be reviewed and versioned
rather than silently updating a benchmark.

Inspection contact sheets are local under `outputs/evaluation_previews/` and
are not distributed. Source PDFs are excluded from package distributions too.

## Evaluation sequence

1. Start with the 16 selected development pages. Include an explicit mapping
   to original document ID and PDF page when generating any subset PDF.
2. Run each backend against the same rendered inputs and record its configuration,
   runtime version, failures, and raw outputs. Increase resolution for handwriting
   only through a documented, matched comparison; do not tune one model using the
   final test results.
3. Manually annotate a manageable sample: text transcription, table cells/spans,
   reading order, and chart labels/values. Keep references outside model inputs.
   Native PDF text is a useful cross-check, not universal ground truth: its reading
   order and character encoding can be wrong.
4. Score text CER/WER, table content/structure, source localization, and chart
   numeric fidelity separately. Report handwritten results as their own category.
   Use a documented human rubric for generated visual descriptions.
5. Add controlled raster/compression, rotation, and blur variants as **derived
   stress tests**, with transformations recorded. They are not independent sources
   or examples of real handwriting.
6. Freeze a separate held-out document/writer set before tuning prompts or
   thresholds further. The pages inspected here are development candidates, not
   an untouched held-out benchmark.
7. Run complete long documents for throughput, memory, completion rate, and
   cross-page behavior after the sampled cases work. Report results by category
   so a long report cannot hide poor handwriting or table performance.

These public documents may overlap model pretraining data; that overlap is
unknown. Do not claim unseen-data performance from their availability alone.

## Coverage still to add

The current handwriting source is one historical writer. It does not establish
performance on modern block handwriting, mathematical notes, Arabic handwriting,
filled forms, or many different writers. A suitable licensed annotated handwriting
dataset or consented examples will be needed for those cases. Invoices, receipts,
Arabic/right-to-left layouts, and low-quality photocopies are additional categories
for the expanded corpus.

## Sources and attribution

- [NASA financial reports](https://www.nasa.gov/budgets-plans-and-reports/agency-financial-reports/).
- [NASA ESTO report](https://esto.nasa.gov/files/AnnualReports/ESTO_Annual_Report_2024.pdf).
- [The AI Scientist, version 3](https://arxiv.org/abs/2408.06292v3), Chris Lu,
  Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha;
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution
  and identify any extracted/modified subsets.
- [NPS diary record](https://npgallery.nps.gov/AssetDetail/539a8c6d-d1c2-40ec-9729-ba874f53ccb1)
  marks the asset public domain and requests credit: Courtesy of North Cascades
  NPS Complex Museum Collection, Aerial Photograph Collection.
- [NASA media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/)
  applies to the NASA reports; individual third-party image credits must be
  reviewed before publishing those images. The corpus records source links and
  keeps original downloads local.
