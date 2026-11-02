# Preserved evaluation artifacts

The three PaddleOCR-VL-1.6 baselines are saved:

1. Original private multimodal PDF: both pages, plus original GPU-run provenance.
2. NASA financial report: first 20 pages of the 127-page source.
3. Handwritten diary: the final five-page selection. Earlier stopped 20-page and
   five-page attempts remain separate, with three completed pages each.

Quantized dots.mocr also has two completed runs of the original two-page PDF,
using separate project and official prompts. The official prompt is the current
default. The GPU batch also attempted NASA 20/20 and diary 5/5 pages: NASA
accepted all 20; diary accepted three and retained two token-limit failures.
See [GPU batch results](quantized-gpu-batch.md).

The failed unquantized CPU experiment remains preserved as historical evidence
in its BF16/CPU archive namespace. Its executable support and weight downloads
were retired; registry entries are under `retired_cases`, excluded from new active
snapshot batches. The working GPU approach remains supported.

Registry: `evaluation/preserved-runs.json`. Local model namespaces:

- `outputs/evaluation_archive/PaddleOCR-VL-1.6_output/`
- `outputs/evaluation_archive/dots.mocr-Q4_K_M-mmproj-Q8_0_output/`

The latest snapshot index is `outputs/evaluation_archive/latest-snapshots.json`,
grouped by model under `models`. Every timestamped snapshot includes a SHA-256
inventory, available source PDF/images/crops, raw predictions, JSON/Markdown/HTML,
logs, timings, source-page mappings, and runtime/code metadata. The NASA archive
also contains a PDF of exactly the 20 evaluation pages and its source-page map.
Dots.mocr snapshots retain the source-derived table checks and comparison JSON.
See [local findings](dots-mocr-local-validation.md) for incomplete visual crops.

Snapshots are append-only. Normal parsing does not overwrite them. Create another
snapshot of one case, or omit `--case` to snapshot all registered cases:

```bash
uv run python scripts/archive_evaluation_runs.py --case handwritten_replacement_5_pages
```

For a future active run, a watcher can wait for inference and exports to finish:

```bash
uv run python scripts/archive_evaluation_runs.py --watch-case <registered-case-id>
```

The watcher also snapshots failed/interrupted terminal runs. It is a local
process; an environment shutdown can stop it. Always check run status and the
archive index before claiming completion. Snapshot metadata distinguishes an
available evaluation subset from a completed full document. Temporary and
uncheckpointed page files are excluded.

Predictions are not verified ground truth. Source-derived assistant spot checks
are identified separately from independent human annotations. Historical runs
were not tied to pinned execution-source revisions; copied code is explicitly
identified as the code at archive time. Raw results and source hashes remain
available independently of future changes.

Archives, private references, and PDFs stay local, ignored by Git and excluded
from package distributions. This is local preservation, not a cloud backup.

The secondary [simple-versus-adaptive RAG study](simple-structural-versus-adaptive-rag.md)
is also preserved: completed metrics and answers, judge calibration and failures,
superseded attempts, a frozen runtime and the paired source review. Its registry
entries point to `outputs/evaluation_archive/gpt-oss-120b_output/`.
The readable comparison and non-private measured summaries are repository-ready source
files under `docs/evaluation/` and `evaluation/results/`; raw artifacts stay local.

The README's public-source table/chart screenshots are curated presentation
assets under `docs/assets/`, with attribution and hashes. They are final-pipeline
review examples, not raw-parser accuracy claims or copies of private documents.

The IRENA/IMF evaluation adds source-reviewed references, measured summaries,
reproducible scoring code and a [failure catalog](irena-imf-source-review.md).
Its final detailed results and browser screenshots are local under
`outputs/quality_evaluations/irena-imf-source-review-02/`. A checksum-inventoried
quality-review bundle preserves the evidence separately from the original
predictions; the registry's `quality_reviews` entry identifies its location.
New enriched-run snapshots include the associated reference and scoring results.
The earlier failed IMF snapshot remains preserved.
