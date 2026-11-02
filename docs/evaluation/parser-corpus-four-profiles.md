# Four-profile comparison across the shared test PDFs

Evaluated all 31 selected pages: NASA (20), the original multimodal sample (2), the historical diary (5), and the author mathematics assignment (4). Four profiles, 124 page attempts. Saved predictions were scored without rerunning or repairing OCR. Missing and incomplete generations remain failures.

## Results per PDF

| Test | PaddleOCR-VL-1.6 | dots Q4/Q8 | Unlimited NF4/FP16 | dots original BF16 | Meaning |
|---|---:|---:|---:|---:|---|
| NASA: printed text | 96.59% | 97.13% | 96.17% | 97.14% | PDF-text-layer token F1; not full-page accuracy |
| Original sample: printed text | 92.11% | 95.64% | 91.35% | 94.10% | PDF-text-layer token F1; not full-page accuracy |
| NASA: objective budget values | 24/24 | 24/24 | 24/24 | 24/24 | Values must match their objective row and requested/operating-plan column |
| NASA: goal-label merged rows | 4/4 | 4/4 | 3/4 | 4/4 | Four source-checked rowspan attributes |
| Original sample: sensor table | 20/20 | 20/20 | 20/20 | 20/20 | 20 cell text/origin/span checks on raw output |
| Original sample: chart data | 0/3 | 0/3 | 0/3 | 0/3 | Three category/value pairs must occur in chart-region output; table duplicates do not count |
| Diary: printed spans | 20/31 | 12/31 | 10/31 | 18/31 | 31 selected source spans; missing pages fail their checks |
| Diary: cursive spans | 2/5 | 3/5 | 1/5 | 3/5 | Only five readable source spans; insufficient for a general cursive ranking |
| Assignment: text and mathematics | 13/37 | 29/37 | 11/37 | 32/37 | Same frozen 37-assertion reference for all four profiles |
| Assignment: truth-table cells | 20/20 | 20/20 | 20/20 | 20/20 | 20 ordered cells, with expected header |

## Content recovery and operational reliability

| Measure | PaddleOCR-VL-1.6 | dots Q4/Q8 | Unlimited NF4/FP16 | dots original BF16 | Meaning |
|---|---:|---:|---:|---:|---|
| Printed text across all 22 text-layer pages | 96.42% | 97.07% | 95.98% | 97.02% | Micro token F1 across 7,113 reference tokens; 9 scan-only pages excluded from this metric |
| Finished page generations | 31/31 | 29/31 | 27/31 | 30/31 | Generation completion, not extraction accuracy |
| Strict canonical pages | 31/31 | 29/31 | Unsupported | 28/31 | Acceptance by each saved pipeline adapter, not a model-independent quality metric |
| Photograph localization | 9/10 | 10/10 | 9/10 | 8/10 | 10 selected source regions, one-to-one matching at IoU >= 0.5; no semantic description score |
| Diagram localization | 6/10 | 6/10 | 10/10 | 10/10 | 10 selected regions, including six Venn sketches; does not score diagram relationships |

## Interpretation

**NASA:** the dots profiles are effectively tied on printed content (97.13% Q4 versus 97.14% BF16). All four recover the 24 checked budget values. Paddle and both dots profiles preserve all four checked goal-label row spans; Unlimited preserves three. On NASA page 11, token F1 is approximately 81.1% for both dots profiles, 70.2% for Paddle and 69.6% for Unlimited, consistent with greater label recovery in the organization-chart page. This is token recovery, not organizational relationship correctness.

**Original multimodal sample:** Q4 has the strongest text-layer recovery and passes both page contracts. BF16 recovers useful text and all checked sensor cells, but neither page passes strict normalization: page 1 omits layout category labels; page 2 includes a footer box that collapses after clipping. All four fail the three chart-region category/value checks. The numbers elsewhere in the sensor table are not credited as chart extraction.

**Historical diary:** Paddle has the best coverage (5/5 pages) and selected printed-span result (20/31), ahead of BF16 (4/5; 18/31). Q4 and Unlimited finish 3/5. This mixed diary selection includes printed calendars and astronomical tables. Five cursive spans are too few for a defensible general handwriting winner; BF16 and Q4 tie at 3/5, Paddle scores 2/5 and Unlimited 1/5.

**Author mathematics:** BF16 is best on the fixed text/math checks (32/37), followed by Q4 (29/37), Paddle (13/37) and Unlimited (11/37). BF16 improves mathematical expressions (15/19 versus Q4 12/19), while both dots profiles tie on prose (17/18). All four recover the truth-table cells.

**Images and diagrams:** Q4 localizes all ten selected photographs. BF16 and Unlimited localize all ten selected diagrams, while Paddle and Q4 localize six. These are localization checks, not judgments of generated image descriptions. No universal image/diagram winner follows from these small samples.

**Decision:** retain Paddle for the broadest observed pipeline completion, Q4 dots as the practical local alternative, and original BF16 dots when handwritten mathematical transcription matters most. The unquantized profile is not uniformly better: its original-sample contract failures matter. Unlimited has useful diagram localization but weaker content/coverage results in this tested execution profile. Do not collapse these tradeoffs into one arbitrary accuracy score.

## Method and limits

- Source PDF hashes and page mappings are checked for each profile. The original Q4 job uses the upstream prompt matching the BF16 input PNGs; the later project-prompt rerun is a separate result.
- All native text on the 22 text-layer pages is scored. Markup and formatting are normalized; token multiplicities penalize repetitions and extra tokens. Token F1 does not measure reading order, semantic truth or full-page OCR accuracy. Raster-only lettering can be absent from the native text reference.
- Exact sequence WER is retained in JSON as a diagnostic. Native PDF reading order differs from rendered reading order, so WER is not the headline ranking.
- Scanned pages and visual/structural content have selected assistant-reviewed annotations, not exhaustive independent human ground truth. Faint/ambiguous cursive remains explicitly excluded.
- References were retrospectively source-reviewed, not held out. Three annotation corrections (Annular spelling and two apostrophe normalizations) were preserved in a separate reference revision and applied equally to every profile. No model output was altered.
- Localization uses manually bounded selected regions, one-to-one matching, IoU 0.5 and a 0.25 sensitivity check. Partial annotations cannot establish precision. A single broad infographic reference is unmatched by all profiles and cannot establish that its separately extracted labels were lost.
- The one bar chart and five cursive spans are too small to establish general chart/cursive rankings. Source-corrected pipeline results and VLM enrichment are excluded from raw OCR credit.
- Baselines used a GTX 1660 Ti (6 GiB); BF16 used a Colab T4. Different runtimes, attention paths, preprocessing and conversion lineage prevent attributing differences solely to quantization.
- IRENA and IMF are separate Paddle pipeline experiments, not part of this four-profile corpus.

## Saved evidence

[Complete metrics and per-page results](../../evaluation/results/parser-corpus-four-profiles-v1.json). Local frozen references, prediction SHA-256 traces, detailed selected checks and the checksum inventory are under `outputs/quality_evaluations/four-profile-corpus-evaluation-03/`. Private source text stays outside Git.

```bash
uv run python scripts/parser_evaluation/evaluate_parser_corpus.py \
  --output-directory outputs/quality_evaluations/four-profile-corpus-evaluation-new
```

Current production selection and measured BF16 GPU peak (8.53 GiB): [model selection](parser-model-selection.md). Historical predictions and scores above are unchanged.
