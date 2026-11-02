# Original dots.mocr BF16 versus three retained parser profiles

The original Hugging Face weights were evaluated on a Colab Tesla T4, alongside
saved PaddleOCR-VL-1.6, dots.mocr Q4_K_M + Q8_0 projector, and Unlimited-OCR 2
NF4 / FP16 vision outputs. No baseline predictions were supplied to the cloud model.
The unquantized run used revision `e539fbb52280393adc081b289ec597430a0f9031`,
CUDA and SDPA, with all 3,039,179,264 parameters recorded as `torch.bfloat16`
and quantization disabled. BF16 storage is verified; native BF16 arithmetic was
not independently verified, and the Torch capability check can include emulation.

## Frozen handwriting evaluation

All four saved outputs were re-scored against the same unchanged reference:
37 selected text/math assertions across the author's four assignment pages, plus
20 ordered truth-table cells. Failed or missing outputs stay in the denominator.
References are assistant-reviewed development checks, not independently annotated
full-document ground truth. No OCR answers were corrected before scoring.

| Profile | Text | Mathematics | Combined assertions | Ordered table cells |
|---|---:|---:|---:|---:|
| Original dots.mocr BF16, Colab T4 | 17/18 | 15/19 | **32/37** | 20/20 |
| dots.mocr Q4_K_M + Q8_0 | 17/18 | 12/19 | **29/37** | 20/20 |
| PaddleOCR-VL-1.6 | 8/18 | 5/19 | **13/37** | 20/20 |
| Unlimited-OCR 2 NF4 / FP16 vision | 9/18 | 2/19 | **11/37** | 20/20 |

Original dots.mocr leads on these selected handwriting checks. The improvement
is in mathematical transcription; prose scores tie the quantized version.
Both dots profiles still miss source expressions. These results do not establish
a general handwriting accuracy rate or a universal winner.

## Completion and output validity are separate from accuracy

The cloud run reused the exact 31 PNG inputs from the retained quantized jobs.
Its 30 finished generations produced 28 pages accepted by the unchanged strict
canonical adapter. The quantized baseline accepted 29/31.

| Input selection | Requested | BF16 finished generations | BF16 accepted | Quantized accepted |
|---|---:|---:|---:|---:|
| Original multimodal sample | 2 | 2 | 0 | 2 |
| NASA financial report | 20 | 20 | 20 | 20 |
| Historical diary | 5 | 4 | 4 | 3 |
| Author mathematics | 4 | 4 | 4 | 4 |

On the original sample, BF16 page 1 omitted category labels throughout its layout;
page 2 included a footer box that collapses after clipping. The diary's second
selected page (original page 5) reached the 300-second generation time limit.
Raw replies are preserved and visible in the demo even when normalization fails.
NASA completion is not a measured NASA accuracy score: no equivalent frozen
content reference covers those full 20 pages. Unlimited-OCR remains an experimental
raw-generation profile rather than a canonical backend.

This is a comparison of execution profiles, not an isolated quantization experiment.
The local baselines used a GTX 1660 Ti with 6 GiB VRAM; the cloud profile used a
T4. Hardware, backend, conversion lineage and some token budgets differ. Do not
attribute every difference to quantization or claim hardware-independent speedups.
IRENA and IMF remain pipeline demonstrations; BF16 dots was not tested on them.

## Saved evidence and reproduction

- [Four-profile metrics and per-assertion results](../../evaluation/results/author-handwritten-mathematics-four-profiles-v1.json)
- [31-page BF16/Q4 operational comparison](../../evaluation/results/dots-mocr-colab-bf16-comparison-v1.json)
- [Cloud notebook](../../notebooks/dots_mocr_unquantized_gpu_comparison.ipynb)
- [Four-profile demo selection](../../evaluation/four-parser-profiles-showcase-cases.json)

Re-evaluate and build a fresh local demo from retained outputs:

```bash
uv run python scripts/parser_evaluation/evaluate_four_parser_profiles.py
uv run python scripts/visualization/build_evaluation_showcase.py \
  --manifest evaluation/four-parser-profiles-showcase-cases.json \
  --include-private --output-directory outputs/evaluation_showcase/four-parser-profiles-new
```

Private source files and local outputs are not shipped in Git. The private build
includes the original regression sample and must not be published as a public bundle.
Earlier three-profile reports and the public demo remain independently preserved.

Current production selection and measured BF16 GPU peak (8.53 GiB): [model selection](parser-model-selection.md). Historical predictions and scores above are unchanged.
