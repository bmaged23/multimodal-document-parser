# OCR evaluation: method, results and reports

[Project overview](../../README.md) · [Model comparison guide](../parsing-evaluation/README.md) ·
[Interactive evaluation report](https://bmaged23.github.io/multimodal-document-parser/evaluation/) ·
[Four-model demo](https://bmaged23.github.io/multimodal-document-parser/?v=5.1)

**dots.mocr is our preferred model family overall for this test corpus.** Original
BF16 offers the strongest balance; quantized Q4_K_M language weights with a Q8_0
vision projector are the production choice for the tested 6 GB GPU. This is a
qualitative assessment across the metrics, not a weighted combined score.

## Overall interpretation

| Rank | Profile | Evidence and tradeoff |
| --- | --- | --- |
| 1 | dots.mocr original BF16 | Strong text recovery, 91.67% visual-region recall and 80.96% row/value matching |
| 2 | dots.mocr Q4/Q8 | Lowest scoped text error rate, 0.67%; weaker table associations than BF16 |
| 3 | PaddleOCR-VL-1.6 | Best row/value matching, 86.58%, and heading detection; weaker text and handwriting recovery |
| 4 | Unlimited-OCR 2 NF4/FP16 vision | Lowest row/value matching, 55.62%; two missing handwriting outputs |

| Task | Best → lowest |
| --- | --- |
| Scoped text recovery | Q4 → BF16 → Unlimited → Paddle |
| Table row/value associations | Paddle → BF16 → Q4 → Unlimited |
| Visual-region detection | BF16 → Q4 = Unlimited → Paddle |

BF16 reached **8.53 GiB peak PyTorch allocation** on Colab T4. The Q4/Q8 profile
ran on an **NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM**, with a sampled device peak of
5,914 MiB in the retained measurements. Allocator and sampled device peaks are
different measurements; neither specifies a minimum GPU capacity. Different
hardware, runtimes and preprocessing prevent attributing all differences to
quantization. [Deployment choice and memory evidence](parser-model-selection.md).

## Corpus and reference creation

| Retained test PDF | Pages | Reference page mapping |
| --- | ---: | --- |
| NASA financial report | 20 | Original PDF pages 1–20 |
| Author-written mathematics | 4 | Four photos ordered by handwritten numbers 1–4 |
| IRENA report | 20 | Original PDF pages 13–32 |
| IMF statistical appendix | 26 | Selected PDF pages 1–26, printed pages 97–122 |
| **Total** | **70** | **280 page/profile slots across four OCR profiles** |

The references contain **750 annotations, 29 complete table grids, 5,246 cells,
48 visual regions and 144 printed chart values**. Text blocks retain source
coordinates, reading order, roles and heading relationships where applicable.
Tables retain cell origins, blank cells, header roles and merged-cell spans.
Images, charts, diagrams, captions, footnotes and uncertain handwriting remain
separately identifiable.

The initial eight-page study used visually checked source text within selected
printed regions and manual table/handwriting drafts. The remaining 62 pages were
transcribed by the assistant from rendered source pages. PDFium rendered those
pages; it did not extract their text. **PyMuPDF was not installed or used.**
OCR predictions and Gemma outputs did not supply reference answers.

The author inspected examples and endorsed the method. These are still
**assistant-created retrospective references**, not independent human ground truth.
No individual approval of all 750 annotations is implied. Unresolved handwriting
and deleted text are excluded from text scoring and recorded in an exclusion
ledger. This covers every page of the retained PDFs, not the remainder of the
longer original publications.

## Evaluation process

1. Freeze the reference dataset and record source PDF/image SHA-256 identities.
2. Read each saved native OCR reply, checking its source identity and hash.
3. Decode native text, table HTML, categories, coordinates and generation status.
   Gemma descriptions, recovered PDF images and repaired pipeline output stay
   outside these scores.
4. Match source regions and native output geometrically, using one-to-one matching
   where required. Match table rows by unique keys as a companion to strict grids.
5. Score all 280 page/profile slots. Missing output remains in the denominator;
   Unlimited's two historical handwriting failures remain visible.
6. Save per-page metrics, per-PDF aggregates, overall aggregates, exclusions,
   detailed checks and an HTML report. Freeze the exact scorer with the results.

There are four execution profiles across three model families. All profiles use
saved outputs for the same source pages; completing a page is not proof of accuracy.
No inference rerun or external judge was needed for this evaluation.

## Measured results

| Metric | Paddle | dots Q4/Q8 | dots BF16 | Unlimited |
| --- | ---: | ---: | ---: | ---: |
| Scoped region CER ↓ | 3.62% | 0.67% | 0.72% | 3.20% |
| Strict grid cell-text match ↑ | 83.55% | 82.39% | 79.87% | 85.09% |
| Label-linked row/value match ↑ | 86.58% | 72.05% | 80.96% | 55.62% |
| Table cell-content multiset F1 ↑ | 91.61% | 90.36% | 90.50% | 87.53% |
| Visual-region recall at IoU ≥ 0.50 ↑ | 81.25% | 85.42% | 91.67% | 85.42% |
| Pages with native layout output | 70/70 | 70/70 | 70/70 | 68/70 |

The [interactive report](https://bmaged23.github.io/multimodal-document-parser/evaluation/)
contains the full metric table, document filters, page-level measurements and
**metric explanations in the rightmost column**. The public viewer exposes 53
allowlisted pages; its evaluation report covers all 70 retained pages. Source
links are enabled only for pages present in that public excerpt.

[Aggregate and per-PDF metrics JSON](../../evaluation/results/manual-reference-full-corpus-v1.json) ·
[Detailed study and scope](full-corpus-source-reference-study.md).

## Metric definitions and limitations

| Metric | What is compared | Interpretation and limit |
| --- | --- | --- |
| Region CER / WER | Reference characters/words against the best contiguous span inside overlapping native text blocks | Lower is better; measures scoped recovery and tolerates segmentation differences. Extra text outside the matched span is not penalized. Do not call 100 − CER overall document accuracy. |
| Text token F1 | Token coverage and extra tokens in geometry-matched native text | Higher is better; ignores order and does not establish semantic correctness. |
| Strict table-cell match | Normalized text at the same row/column origin | Different valid table layouts can score poorly; inspect row associations alongside it. |
| Label-linked row/value match | Unique country, objective ID or other row key followed by its ordered values | Tolerates header offsets and objective IDs joined to descriptions. Missing/duplicate output keys fail. Nonunique reference keys are excluded. |
| Cell-content multiset F1 | Multiset of nonempty cell texts | Measures content coverage without validating relationships between rows and values. |
| Merged-cell spans | Row-span and column-span attributes at exact origins | Layout-sensitive; complements cell text. |
| Header/body roles | Explicit native header/body flags at exact origins | Includes body cells and can be dominated by them; no inferred pipeline roles are added. |
| Visual-region recall | Unique OCR image/chart/diagram regions at IoU ≥ 0.50 | Detection and crop coverage, not image interpretation. Logos/decoration are outside scope; false-positive precision is not claimed. |
| Matched crop overlap | Mean IoU among matched visual regions | Read with recall: this average alone does not penalize missing regions. |
| Heading detection | Native heading/title boxes at IoU ≥ 0.30 | Does not score heading levels or parent links absent from native outputs. |
| Matched reading order | Correct order among geometrically matched source text pairs | Unmatched blocks are outside this denominator; pair counts are saved. |
| Printed chart-number recovery | Printed numeric tokens in native text within the chart region | Not series/category accuracy; axis numbers can create matches. Figure crops retained without transcribed values score zero on this OCR-only diagnostic. |

Overall rates pool their underlying units; per-page macro averages give each
scorable page equal weight and are saved separately. Unavailable metrics stay
null rather than becoming zero. There is no weighted overall composite.

Normalization equates whitespace, quote/dash variants, solid-dot glyph variants
and spaced ellipses. It preserves dots versus stars and positive versus negative
numbers. Complete mathematical LaTeX equivalence is not implemented; formula
semantic accuracy is not claimed. Older handwriting presence checks and repaired
pipeline scores use different references and must not be merged into this study.

## Output artifacts and reproduction

| Artifact | Contents | Public availability |
| --- | --- | --- |
| `metrics.json` | Overall/per-PDF metrics, denominators, scope and hashes | Repository summary and demo report |
| `page-metrics.json` | All 280 page/profile metric rows | Demo report download |
| `excluded-references.json` | Annotation identifiers and exclusion reasons | Demo report download |
| `failure-traces.json` | Expected/actual source text, cell checks and row associations | Local; contains private source content |
| `frozen-reference.json` | Exact reference version used | Local; contains source transcriptions |
| `frozen-scorer.py` | Exact scorer bytes matching the recorded SHA-256 | Local reproducibility artifact |
| `index.html` | Filterable comparison and metric explanations | Published demo evaluation report |

With the retained private corpus and native evidence packages available, run from
the repository root using the project uv environment:

```bash
uv sync --locked --group dev
uv run python scripts/parser_evaluation/evaluate_manual_reference_corpus.py \
  --reference outputs/reference_review/full-corpus-source-review-06/dataset.json \
  --output-directory outputs/quality_evaluations/new-reference-evaluation
uv run python scripts/visualization/build_manual_reference_report.py \
  outputs/quality_evaluations/new-reference-evaluation
uv run python -m http.server 8772 --bind 127.0.0.1 \
  --directory outputs/quality_evaluations/new-reference-evaluation
```

Open <http://localhost:8772/>. A fresh clone includes the scorer and public metric
summary, but private references, PDFs and native runs must be supplied to reproduce
scores. The scoring command requires a fresh output directory and preserves earlier runs.

The metric tests compare edit distance with independent dynamic programming and
assignment results with exhaustive optima. Regression checks cover solid-dot and
ellipsis equivalence, joined objective IDs, sign/symbol preservation and repeated
values. Browser checks cover report filters, all 70 reference pages and all 280
metric rows. Dataset, source and scorer hashes make the frozen run identifiable.

## Separate studies

- [Earlier selected handwriting assertions](author-handwritten-mathematics.md).
- [Earlier 31-page comparison](parser-corpus-four-profiles.md).
- [Pipeline repairs and final source checks](final-source-quality-repair.md).
- [Visual-model comparison limits](five-model-vlm-benchmark.md).
- [Secondary chunking/RAG study](../chunking-and-rag/README.md).

The new raw-OCR comparison does not replace these historical artifacts or give
raw models credit for Gemma-generated or source-repaired output.
