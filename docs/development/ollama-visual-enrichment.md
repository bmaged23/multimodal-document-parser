# Visual enrichment setup

## Default: Gemma 4 31B on Ollama Cloud

Use the existing project uv environment. Set `OLLAMA_API_KEY` in your environment
or ignored project `.env` (never commit the key). No local Ollama server or model
weights are needed for cloud inference.

```bash
uv run document-parser enrich outputs/parser_runs/paddleocr-vl-1.6-reviewed \
  --provider ollama-cloud --model gemma4:31b \
  --output-directory outputs/parser_runs/my-gemma-enriched-run
```

For the original three-visual validation and source-derived checks:

```bash
uv run python scripts/visual_evaluation/run_visual_enrichment_validation.py \
  --provider ollama-cloud --model gemma4:31b --suffix NEW
```

Use a fresh suffix/destination. Cloud receives the target crop and page-context
images. Requests go only to the official Ollama HTTPS origin with the key loaded
privately from `.env` or environment. Authenticated redirects are disabled.
Structured JSON is requested via the prompt; local validation enforces the schema.
Native output is preserved before any complete JSON code fence is unwrapped.
Credentials and authentication headers are never copied into run artifacts.
Cloud model aliases/catalog digests do not guarantee immutable deployment weights.
See [actual validation](../evaluation/gemma4-cloud-visual-validation.md).

The optional Qwen Cloud provider uses separate `DASHSCOPE_API_KEY` credentials and
an official regional endpoint. It is not needed for the selected Ollama model.
Its software transport is tested, but no real Qwen Cloud inference has run.

## Historical local Ollama reproduction

Milestone 5 adds a separate stage after OCR. Use the existing project uv environment:

```bash
uv sync --locked --group dev
uv run python scripts/model_runtimes/setup_ollama_runtime.py --model qwen3-vl:2b-instruct
uv run document-parser enrich outputs/parser_runs/paddleocr-vl-1.6-reviewed \
  --provider ollama --model qwen3-vl:2b-instruct \
  --output-directory outputs/parser_runs/paddleocr-qwen3-vl-visual-enrichment-01
```

Setup downloads a checksummed portable Ollama v0.40.13 release under `.cache/ollama/`,
starts a server bound to 127.0.0.1:11435. Models are pulled only with an explicit
`--model` argument and kept in the same ignored cache. The Qwen3.5 4B weights were
removed at the user's request and are not downloaded by default. `zstandard` is a development/bootstrap dependency in the existing
uv environment, not another virtual environment. No system service or global
installation is required. Port 11435 is dedicated to this project. Stop any OCR
GPU processes first. The enrichment stage runs one request at a time and requests
model unloading in `finally`, including after failures. Check `/api/ps` if unloading
was not confirmed. Setup downloads require internet; inference sends images only
to the explicitly configured Ollama endpoint (loopback by default).

Qwen3-VL 2B and Qwen3.5 4B historical experiments remain preserved separately.
The current CLI default is Gemma 4 31B on Ollama Cloud.
Local reproduction requires the explicit `--provider ollama` flags shown above.
See README for cloud credentials and the regional endpoint. The cloud transport
uses JSON Object mode, includes the schema in the prompt, validates locally,
and preserves the native response, returned alias and token usage. No hosted
checkpoint revision or GPU allocation is invented. API keys stay outside outputs.

The model tag is convenient for setup but mutable. Each run captures its exact
Ollama digest, server version, options and prompt SHA-256. Raw responses, source
region mappings and exact input images are saved per element. Runtime SHA-256
and model download metadata live under `.cache/ollama/`.

## Inputs and output

Only fully parsed canonical documents may be enriched. The input is copied to a
fresh destination; earlier extraction and archives remain unchanged. Existing
visual descriptions are rejected to avoid replacing reviewed content. The model
receives the visual crop plus the full page (a red rectangle marks the target),
resized to at most 2048 and 1536 pixels respectively. Explicit chart crops are
re-rendered from the identical PDF at 300 DPI before resizing. This supplies context when
OCR boxes cut through chart axes or diagram branches, but does not correct the
original predicted box. Full-page context includes nearby text and tables;
this is not a crop-only OCR accuracy experiment.

Descriptions are optional fields on existing image/chart elements. Each carries
`method=generated`, exact model identity, and structured `analysis`: classification,
title, axes, numeric series, flowchart nodes/edges and uncertainty. OCR type and
provenance are preserved. For an image that the VLM identifies as a chart, the
classification and series stay inside its generated analysis. Existing extracted
chart series are preserved. Optional schema additions retain version 1.0 and
backward readability; the checked-in JSON Schema is regenerated.

Numeric points require explicit `printed_label`, `visual_estimate`, or `unknown`
provenance. This stage disables estimates: numeric data must claim a printed label
or remain null/unknown. These labels are model claims, not external verification;
the prompt forbids calculating heights or copying nearby table values. Generated
content can still be wrong and must be checked against the source. Descriptions
and numeric claims are never promoted to independently verified OCR or ground truth.

The final quality pass adds one bounded chart-review request and retains both
draft and review responses. Native PDF checks can normalize an explicitly defined
unit and correlate endpoints for narrowly supported vector line charts. Those
numeric records use separate extracted provenance; source-based endpoint summaries
are labelled separately from VLM prose. Unsupported layouts abstain and warnings
remain visible. See [final source repairs](../evaluation/final-source-quality-repair.md).
Use `scripts/parser_evaluation/finalize_saved_visual_quality.py` to apply those source-only checks
to an existing saved run without new OCR or VLM calls. The
[five-VLM comparison](../evaluation/five-model-vlm-benchmark.md) scores raw model
responses and excludes these repairs.

The parser's explicit chart type can select the chart schema. Other visuals use
target-only model classification; preceding headings are deliberately ignored
because a missing OCR heading can incorrectly route a later raster image. A
type-specific response schema forces irrelevant fields to be empty and non-chart
axes to be null. Classification results, effective prompts and schemas are retained.

Flowchart IDs must be unique and every edge must reference existing nodes.
Non-chart results cannot contain numeric series. Non-flowchart results cannot
contain nodes or edges. Invalid JSON, schema violations, forbidden estimates,
truncated responses and transport failures are preserved as per-element failures;
other visuals continue. Interrupted runs are labelled interrupted. No silent retry
or model JSON repair is performed.

`enrichment-manifest.json` records progress and partial completion separately from
historical OCR `run-status.json`. The resulting document JSON, Markdown and HTML
include generated descriptions, values/bases, connections and uncertainty. The
same review keeps the PDF on the left, output on the right, source links, page
navigation, Markdown source and JSON views. Untrusted model strings are escaped
and Markdown HTML is sanitized. Raw inputs/responses remain downloadable locally.

## Validation scope

Synthetic integration checks cover failure isolation, raw-response/input retention,
input hashes, generated versus extracted provenance, graph invariants, exports and
model unloading. They verify software behavior, not real model accuracy. The first
real smoke test uses the original two-page PDF's chart, flowchart and raster image.
Source-derived numeric and graph assertions must be reported separately from full
accuracy evaluation. Private PDFs, weights and run artifacts stay ignored by Git.

Official documentation: [Ollama generate API](https://docs.ollama.com/api/generate),
[Qwen3-VL model](https://ollama.com/library/qwen3-vl:2b-instruct).
