# PaddleOCR parsing and HTML review

## Run the pipeline

From the repository root:

```bash
uv sync --locked
uv sync --project backends/paddleocr_runtime --locked
uv run document-parser prepare data/input_documents/example.pdf
uv run document-parser parse outputs/parsed_documents/<document-hash>-dpi-144
```

Use `--output-directory outputs/parser_runs/my-first-run` to choose a descriptive
run directory. It must not already exist. The preparation bundle stays intact.
`--runtime-directory` selects the isolated runtime directory; `--device` selects
`gpu:0` (default) or `cpu`.

A completed run contains:

```text
review.html                      Side-by-side HTML review
document.json                    Canonical parsed document
document.md                      Readable extraction with source references
source.pdf                       Original PDF snapshot
assets/pages/                    Source page images
assets/crops/                    Chart/image regions
raw/paddleocr/page_0001.json      Original backend output
raw/paddleocr/inference-manifest.json
                                 Version, timing, and sampled device-memory data
inference.log                    Backend diagnostics
normalization-notes.json         Heuristics and unsupported-label notes
run-status.json                  Completed/failed status and run mode
run-summary.json                 Element counts and copied inference measurements
```

Failed runs retain their logs, partial raw results, and a failed run status for
diagnosis. Do not interpret a failed run's artifacts as a complete extraction.

## Review the output

Open `review.html` in a desktop browser, keeping it beside its asset directories.
For consistent browser PDF support, serve only the run directory locally:

```bash
uv run python -m http.server 8765 --bind 127.0.0.1 --directory outputs/parser_runs/my-first-run
```

Then visit `http://localhost:8765/review.html`.

- Left: original PDF, a page-number field with Go/previous/next controls, and a
  rendered-page mode. Go updates the source and parsed output together.
  Use PDF page numbers, which may differ from printed page labels.
- Right: rendered Markdown preview (including extracted images), Markdown source
  code, structured content cards, and JSON tabs. The current-page filter applies
  to all four views. Clear it to inspect the full document; downloads always
  contain the full document. Per-page JSON is a page view, not a complete canonical
  Document object.
- Source buttons switch to the correct rendered page and highlight the source box.
- Search filters cards and page views. The current-page checkbox narrows output
  to the selected page; the page number field synchronizes both panels.
- Download links expose the Markdown and JSON files.
- Model text is escaped. Markdown previews are rendered with markdown-it-py and
  sanitized with nh3 before embedding. Tables are reconstructed from validated cells, never
  inserted as arbitrary backend HTML. No external fonts, scripts, or analytics load.
- Formulas currently display their extracted LaTeX rather than typeset math.
- Image descriptions and structured chart numbers belong to the enrichment milestone.

## Normalization choices

PaddleOCR's `parsing_res_list` order is retained. Pixel boxes become normalized
coordinates on the rendered page. Out-of-bounds boxes are clipped with notes;
invalid/degenerate boxes fail. Heading levels use backend labels or leading
Markdown hashes and are explicitly marked as inferred. They are not yet a
validated document hierarchy benchmark.

HTML tables retain row/column spans and header flags. Missing/malformed table
structures fail the run rather than silently discarding cells. Unknown labels
are retained as text with a note; their original data remains in raw output.
Visual regions are cropped from source page images. Any backend-produced chart
content remains available in the raw JSON until chart normalization is implemented.

## Import existing results

```bash
uv run document-parser parse <prepared-bundle> \
  --raw-results-directory <saved-page-results> \
  --output-directory outputs/parser_runs/imported-results
```

Saved results must be named `page_0001.json`, etc., one per prepared page, with
matching `width`/`height` and a `parsing_res_list`. Both direct result dictionaries
and `{"res": {...}}` wrappers are accepted. Only import results produced from
those exact prepared page images without orientation/unwarping transforms.
Imported results are marked `mode=import`; no live inference is performed.

## Verification

```bash
uv run pytest tests/unit tests/integration
uv sync --locked --group browser
uv run --group browser playwright install chromium
uv run --group browser pytest tests/browser
```

Browser tests exercise page navigation, source highlights, output tabs, search,
escaping, and a narrow viewport. CI has separate software and browser jobs.
Synthetic result fixtures are explicitly labeled and never count as real model
quality evaluation. GPU run measurements are operational observations until a
separate annotated reference dataset and scoring procedure are established.


## Resume and inspect long runs

The inference runtime checkpoints each completed page and its measurements.
Completed raw files are retained when a job is interrupted. To resume a stopped
run using the original one-model, one-page configuration:

```bash
uv run python scripts/resume_parsing_run.py <prepared-bundle> <existing-run-directory>
```

This validates input identity and saved page dimensions before reusing results.
The runtime skips checkpointed pages and retains the previous manifest as history.
The supervisor refreshes JSON, Markdown, and HTML after completed pages. A document
with `status=parsing` is explicitly partial; `status=parsed` is written only after
all expected raw page results are available. The HTML displays a progress banner
and reads the local run status. Refresh the page to load newly exported content.

`run-status.json` reports completed/total pages; `supervisor.log` and
`inference.log` retain diagnostics. Long jobs launched from an interactive tool
session can stop when that session disappears. A detached process launch avoids
ordinary chat/session interruption; an environment shutdown still requires a
checkpoint restart. Never infer worker liveness solely from an old running flag.

The optional `document-parser parse --page-window 2` permits bounded multi-page
input through one shared model. The default remains 1. The incomplete shadow test
does not justify a throughput or accuracy claim for the queued option.
