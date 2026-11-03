# Compare four OCR profiles in one parsing flow

The demo uses one parsing flow with different OCR runners: prepare the PDF,
normalize each runner's own predictions, enrich its detected visual regions with
Gemma through Ollama Cloud, and display the saved outputs beside the source.
The production container's Q4 default is a hardware choice for the 6 GB laptop;
it is not a claim that quantized dots has the best quality.

## Scope

| Document | Full saved selection | Public excerpt |
| --- | ---: | ---: |
| NASA financial report | 20 pages | 18 pages |
| Author handwritten mathematics | 4 pages | 4 pages |
| IRENA charts and tables | 20 pages | 15 pages |
| IMF statistical tables | 26 pages | 16 pages |

The complete matrix contains 70 source pages and four execution profiles. The
public viewer uses the same 53-page excerpt for every profile. Separately credited
third-party photos and excluded narrative pages remain outside the published
bundle; see the [source/page allowlist](../../evaluation/public-demo-selection.json).

The profiles are PaddleOCR-VL-1.6, dots.mocr Q4_K_M language/Q8_0 projector,
original dots.mocr BF16 on Colab T4, and Unlimited-OCR 2 NF4 language/FP16 vision.
They are four execution profiles across three model families. Hardware,
quantization, processors and output limits differ; this is not a controlled causal
quantization experiment. Completion and output validity are not accuracy.

## Read the outputs

Choose a document and its **Model**. Enable **Compare models** to switch between
two outputs or show them together. The original-page control keeps source and
output synchronized, and **Expand PDF + output** enlarges the reader.

Each structured OCR card names its OCR profile at the top right. Generated visual
descriptions appear in separate **Gemma** cards, with raw initial/review replies
available for inspection. Model review, source correlations and corrections stay
visible; generated descriptions and numeric values are not independent ground truth.
The raw OCR reply remains separately available. A page rejected by normalization
shows its retained raw reply rather than claiming that an empty page was accurate.

There is one set of model choices for every document, rather than a separate
pipeline-stage selector. Historical Paddle source-reviewed evaluations remain
separate evidence; they are not relabeled as outputs from another OCR model.

## Reuse and provenance

Saved NASA and mathematics OCR runs are reused for all four profiles. Saved
Paddle IRENA/IMF OCR runs are also reused. The other three profiles receive new
IRENA/IMF inference: 46 pages per profile. Existing compatible Paddle/Gemma output
is reused; other models supply their own visual boxes and crops, never Paddle's
boxes. Completed identical Gemma requests can be reused only when the full request
and provider catalog digest match. Raw replies and original provider timings stay
intact, so cache hits must not be counted as new model latency measurements.

Unlimited native blocks receive a separate documented normalization step. Their
0–1000 coordinates and original HTML table spans are retained; invalid pages are
rejected atomically. A title's heading level is inferred and disclosed. Raw native
outputs remain available even when canonical normalization fails.

Full-precision dots stores all recorded parameters in BF16 and runs on GPU; this
does not establish native BF16 arithmetic on T4. Its observed allocator peak is
not total device VRAM or a minimum capacity. The earlier frozen quality reports
remain historical measurements, not scores for this expanded manual comparison.

## Run from retained artifacts

Private source selections and full saved model outputs are not included in a
fresh clone. These commands operate on the retained local evaluation corpus:

```bash
uv run --no-sync python scripts/parser_evaluation/run_demo_model_matrix.py prepare
uv run --no-sync python scripts/parser_evaluation/run_demo_model_matrix.py dots
# Finish dots before loading another model on the same GPU.
UNLIMITED_OCR_CACHE_DIR=/path/on/a/large/disk uv run --no-sync python \
  scripts/parser_evaluation/run_demo_model_matrix.py unlimited --wait-for-dots
uv run --no-sync python scripts/parser_evaluation/run_demo_model_matrix.py prepare-cloud
uv run --no-sync python scripts/visual_evaluation/enrich_demo_model_matrix.py --watch
```

Use the [pinned Colab notebook](../../notebooks/complete_four_model_demo_bf16.ipynb)
for the two missing BF16 selections. It checks the private matched-input ZIP,
records dtype/GPU/hash evidence, and downloads its results. Never publish that
private input ZIP. Import and preserve the result using
`scripts/parser_evaluation/watch_demo_cloud_results.py`.

The release builder refuses an unfinished 4×4 matrix, mismatched PDF identities,
changed OCR text/tables or unresolved visual failures:

```bash
uv run --no-sync python scripts/visualization/build_four_model_demo.py \
  --output-directory outputs/evaluation_showcase/four-model-public-new
```

Keep full archives under their exact model namespaces. Publish only the curated
`public-demo` bundle and its checksum, then validate the viewer with
`scripts/visualization/verify_four_model_demo.py`. The demo is static saved output;
it does not deploy a live inference API or rerun the models when opened.
