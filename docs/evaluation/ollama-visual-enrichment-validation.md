# Milestone 5 local visual validation

The original two-page synthetic PDF was enriched from preserved PaddleOCR-VL-1.6
extraction with local Ollama v0.40.13 and `qwen3-vl:2b-instruct`. Exact model digest:
`afc393e3572c07db0bc84aa460d928a16c75f3fbb9ca9e0dccc4fbd2dafb573e`.
The API reports Q4_K_M quantization. The portable runtime archive SHA-256 is pinned
in `scripts/setup_ollama_runtime.py`. Existing project uv `.venv` is used; no extra
Python environment or global Ollama installation was created.

## Actual observations

Final development run: `outputs/parser_runs/paddleocr-qwen3-vl-visual-enrichment-04/`.
The three target visual interpretations completed normally. Selected assistant
source-derived checks passed 6/6: 18.4, 23.1 and 31.7 ms at the three humidity
categories, plus the three flowchart connections including YES/NO branch labels.
The synthetic microscopy description noted circular outlines and the printed
50 um scale bar. That description has not received a full independent annotation.

The API's post-generation loaded-model observations report `size_vram=2001668668`
bytes and context length 4096 on this GTX 1660 Ti setup. This is an Ollama model
allocation observation, not peak GPU memory and not directly comparable with
previous total-device samples or Torch allocator statistics. Model unloading is
requested after the stage; raw response duration/token counts are preserved.

These six assertions establish only those selected facts on a development sample.
They are not CER/WER, a general VLM benchmark, or independent human verification.
The reference was not fed to the model, but full-page context necessarily contains
nearby text/table values; the prompt forbids copying those into chart values.

## Failures and remaining limitations

Four development attempts are preserved independently. Run 01 recovered the values
but omitted series units and repeated uncertainty text. Run 02 failed semantic
validation for all three outputs. Run 03 used generated type classification and
misclassified the chart as an image, losing its structured numeric extraction.
Run 04 uses the explicit extracted chart type and preceding flowchart/image
headings to choose the output schema. Unlabelled visuals use generated classification.
These source hints may themselves be incorrect and remain recorded in provenance.
There is no silent retry or output repair. The final default has type-specific
empty fields and prevents axes on non-chart requests; run 04's earlier raw response
is retained unchanged and predates that last schema constraint.

Even run 04 is imperfect: it splits one measured quantity into three series named
by humidity, calls the chart-side explanation a legend, and emits irrelevant
flowchart uncertainty statements about another figure. It also supplies conceptual
flowchart axes despite no printed axes; future requests now forbid non-chart axes.
The selected checks do not score these errors. Generated numeric/graph content
and uncertainty are predictions, with source links and exact model provenance,
not verified extracted facts. A larger held-out set and better models or reviewed
corrections are future evaluation work.

## Preservation and inspection

JSON/Markdown/HTML contain the descriptions, numeric values and bases, nodes,
connections and uncertainty. Each raw response, classification response (if used),
request schema, effective prompt, image bytes/hashes and source-region mapping is
preserved. Existing OCR text, tables and source crops stay unchanged. Generated
chart series have separate `series_provenance`; the original element provenance
still identifies OCR. A post-generation export added this metadata without changing
predictions. Both before/after immutable snapshots are kept.

All runs live in the combined namespace
`outputs/evaluation_archive/PaddleOCR-VL-1.6+Qwen3-VL-2B-instruct_output/`.
Check `enrichment-manifest.json` separately from historical OCR `run-status.json`;
an archive containing two completed OCR pages does not imply three successful VLM
interpretations. Private source PDFs and model artifacts remain ignored by Git.

See [setup and reproduction](../development/ollama-visual-enrichment.md).

## Correction after reviewing the raster-image response

The six numeric/graph assertions did not cover the third visual's type. Run 04
incorrectly routed the microscopy image to a flowchart task: Paddle labelled its
section title as ordinary text, so a preceding-heading heuristic inherited the
earlier flowchart title. The later raw image response describes that unrelated
flowchart. The earlier statement about its 50 um scale bar applies to run 01, not
run 04. Run 04 is not a successful three-visual quality validation.

The heading heuristic has been removed. Only an explicit canonical chart type
provides a routing hint; other visuals use target-only model classification.
Milestone 5 quality validation remains pending. The next evaluation must include
visual-type assertions for all three regions as well as numbers and connections.
The preserved raw predictions and snapshots remain unchanged. This software error
must be distinguished from the small model's independent grouping/uncertainty errors.
