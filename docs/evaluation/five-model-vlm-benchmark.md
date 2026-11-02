# Five VLMs on five varied source visuals

The benchmark is prepared and access-tested on 2026-10-31. **A five-model quality
comparison cannot yet be completed:** the configured Ollama account returned
HTTP 402 Payment Required for all four alternative models. Gemma's five cached
cases were scored. There is no defensible winner while the alternatives have no
quality results.

### Two-key cloud access recheck, 2026-10-31

A fresh bounded image request using each Drive credential produced different
results. `Ollama API key.txt` returned HTTP 429 for `gemma4:31b`, explicitly citing
the monthly usage limit. Further generation probes on that key were skipped.
`Ollama API key2.txt` successfully described the saved equipment photograph using
`gemma4:31b` (HTTP 200, 19 output tokens). The same second key returned HTTP 402
for GLM, DeepSeek, Kimi and Mistral from the shortlist. Those responses explicitly
state that the requested model is excluded from free usage and recommend Gemma
as included in the Free plan.

This live evidence confirms some free Gemma access despite the public pricing
page's general paid-token wording. It does not establish five free vision
models, or that the first key is invalid. The configured credential remains the
first key; this access check did not switch `.env`, restart parsing, or run local
inference. Successful access is not a quality score. Private, secret-free probe
results are retained under
`outputs/quality_evaluations/ollama-two-key-cloud-recheck-20261031T163453Z/`.

At the user's subsequent request, the project configuration was switched to
`Ollama API key2.txt` and the effective client's credential was verified without
logging it. The four alternatives were retried using the original benchmark's
full saved chart crop, context image, prompt and schema. Each again returned
HTTP 402 on its first chart request; remaining cases were skipped. Gemma was
excluded entirely. The active credential is now the second key. Configuration:
`evaluation/vlm_benchmarks/irena-four-cloud-alternatives-v1.json`; fresh evidence:
`outputs/quality_evaluations/ollama-four-cloud-alternatives-key2-01/`.

A subsequent search used the live authenticated cloud catalog rather than local
library listings. Eight additional catalog-listed vision models were probed with
the second key and a saved photograph, with at most 32 output tokens requested:
`minimax-m3.1-flash`, `mimo-v2.6-flash`, `muse-spark-open`, `kimi-k2.6`,
`minimax-m3`, `mimo-v2.6-pro`, `kimi-k2.7-code`, and `mistral-large-3:675b`.
All eight returned HTTP 402 explicitly excluding the model from free usage and
recommending Gemma. These models are hosted, but are not accessible under the
tested key's free allowance. `clef:27b` advertises vision and decision capability
and was not treated as a general chat description model. No additional free
general vision model was found. Catalog metadata and access evidence are saved
under `outputs/quality_evaluations/ollama-hosted-alternatives-key2-20261031T164457Z/`.

The user selected `gemma4:31b` on Ollama Cloud for continued use with the second
key. It remains the existing enrichment default based on confirmed access;
this decision does not imply that it won the blocked multi-model comparison.
No parsing rerun is needed to retain this configuration.

## Fixed shortlist

| Exact Ollama Cloud API model | Selection rationale | Current result |
| --- | --- | --- |
| `gemma4:31b` | Existing baseline; small deployment cost compared with the larger contenders | Five cached cases evaluated |
| `glm-5.3-flash` | Native vision; model card reports document/chart evaluations | HTTP 402 |
| `deepseek-v4.1-flash` | Native vision; promising general visual reasoning candidate | HTTP 402 |
| `kimi-k3` | Native multimodal model; candidate for detailed visual interpretation | HTTP 402 |
| `mistral-large-4` | Native multimodal model from another model family | HTTP 402 |

These are promising candidates, not a pre-existing ranking. All five appear in
the live authenticated cloud catalog with vision capability. Library listings
and catalog visibility do not establish that the account can run a model.
Official model cards: [Gemma 4](https://ollama.com/library/gemma4),
[GLM](https://ollama.com/library/glm-5.3-flash),
[DeepSeek](https://ollama.com/library/deepseek-v4.1-flash),
[Kimi](https://ollama.com/library/kimi-k3),
[Mistral](https://ollama.com/library/mistral-large-4).

Ollama's [pricing page](https://ollama.com/pricing) explains pay-as-you-go usage
credits. No subscription purchase, credit purchase, account change or local
model download was performed.

## Five source regions

All cases come from the same IRENA selection, with SHA-256 identity and original
page mapping retained. This is a small development comparison, not a broad
benchmark across document publishers or languages.

| Case | Sample / original PDF page | What it tests |
| --- | --- | --- |
| Dense multi-series line chart, Figure S1 | 4 / 16 | Crowded endpoint labels, legend-color association, seven technologies, 14 printed values |
| Grouped bar chart, Figure 1.1 | 15 / 27 | Region/year categories, large and small values, 18 printed values, GW defined in a note |
| Unlabelled scatter plot, Figure S6 | 9 / 21 | Axes/units, qualitative relationship, abstaining from invented numerical points or axis-tick categories |
| Equipment close-up | 13 / 25 | Open turbine nacelle, suspended component, visible person and source-grounded detail |
| Landscape photograph | 20 / 32 | Dam, water/spray, structures and mountainous surroundings |

The three chart types are deliberately different. The two photographs contrast
close-up machinery with a wide landscape, rather than repeating the same scene.

## Protocol

- Every model receives the **same saved image bytes, prompts and JSON schema**,
  using 300-DPI-rendered chart crops limited to 2,048 pixels and a 1,536-pixel
  page-context image. Input hashes are checked before use.
- Requested temperature is 0 and output budget is 4,096 tokens. Effective
  provider reasoning behavior can vary by model and must not be assumed equal.
- Visual kind is supplied; this tests description/extraction, not classification.
- Every chart has a draft and one review request. The review receives that
  model's own draft and native source text, never expected-answer annotations.
- Each photograph has one description request. Gemma's matching saved responses
  are reused instead of incurring new inference.
- **Score raw draft and raw review responses separately.** Native-vector repairs,
  source-derived summaries and other postprocessing are excluded from VLM scores.
- Expected answers are opened after inference. Chart checks require the correct
  series/category/value association, preserve signs, score units separately and
  assess numeric abstention. Explicit unit synonyms such as `gigawatt`/`GW` are
  equivalent under a documented normalization rule.
- Photo coverage and unsupported claims are reviewed against the source images.
  Keyword presence alone is not a factuality score. Reviewer identity and limits
  are retained; there is no independent human adjudication.
- HTTP 401/402/403 stops further calls to that model. Unavailable cases have no
  quality score. They are never counted as wrong answers or ranked below working
  models on accuracy.

With all models accessible, each new model needs eight requests: six for the
three chart draft/review pairs and two for the photographs. There is no OCR
rerun, model download or GPU inference in this comparison.

## Results currently available

| Gemma 4 property | Draft | Review |
| --- | --- | --- |
| Dense line chart printed values | **5/14** | **6/14** |
| Grouped bar printed values | 18/18 | 18/18 |
| Total printed-value assertions | 23/32 | 24/32 |
| Units, with explicit synonym normalization | 8/8 | 8/8 |
| Scatter numeric abstention | Pass | Pass |
| No invented scatter categories | Pass | Pass |

Both photographs cover 4/4 selected visible features. The equipment caption's
claim that the component is "being lowered" adds a motion direction not
established by a still photograph; this is recorded as a minor unsupported claim.
The line-chart review also incorrectly describes all technologies as declining,
although three series have increasing source endpoints. Structured numerical
accuracy and prose factuality therefore both matter.

These observations explain why a larger general-purpose VLM must be measured on
the actual task, and why a successful JSON response is not enough. The pipeline's
separate [final source-based repairs](final-source-quality-repair.md) are not
credited to Gemma in this comparison.

All four alternatives failed their first actual chart request with HTTP 402.
Their remaining four cases were skipped to avoid repeating an account-level
failure. The 25-cell matrix therefore contains **five completed Gemma cases,
four access failures and sixteen cases not attempted because of those failures**.
No multi-model quality ranking, latency ranking or winner is published.

## Reproduce once account access is available

Configuration: `evaluation/vlm_benchmarks/irena-five-visuals-v1.json`.
Source assertions and photo rubric:
`evaluation/vlm_benchmarks/irena-five-visuals-reference-v1.json`.
Manual observations:
`evaluation/vlm_benchmarks/irena-five-visuals-manual-review-v1.json`.

```bash
uv run python scripts/run_vlm_benchmark.py \
  evaluation/vlm_benchmarks/irena-five-visuals-v1.json \
  --output-directory outputs/quality_evaluations/ollama-five-models-five-visuals-02
```

Use a new output directory; earlier results are preserved. The scorer and runner
reuse saved source assets and do not alter the parser baselines. Local results,
model requests/responses, account statuses and HTML comparison are under
`outputs/quality_evaluations/ollama-five-models-five-visuals-01/`.

Once all five models have comparable outputs, finish manual photo and chart-prose
review, compare numerical fidelity and unsupported claims first, then operational
failure rate and measured response time. Retain separate chart and photo results;
a single model need not be best at both. Any recommendation is limited to these
five cases; additional held-out cases are needed for a general claim.
