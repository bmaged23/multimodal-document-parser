# Quantized GPU evaluation batch

This batch tests the retained dots.mocr Q4_K_M language model and Q8_0 projector
on the first 20 NASA financial-report pages, then the same five diary pages used
for the final PaddleOCR baseline (original diary PDF pages 4, 5, 7, 6, 74).

```bash
uv run python scripts/run_quantized_evaluation_batch.py --suffix 01
```

The script requires the existing local corpus/preparation bundles and the built
CUDA 12.6 native runtime. It does not download models or recreate the retired CPU
approach. A fresh suffix creates separate outputs and never overwrites a previous
batch. Progress is saved after every attempted page in
`outputs/parser_runs/dots-mocr-q4-evaluation-batch-01.json`.

## Execution and limits

One shared GPU model handles requests sequentially. Each source page is imported
into a single-page PDF, prepared at its original case DPI, and checked for exact
RGB pixel equivalence before inference. Source DPI is 144 for NASA and 300 for the
diary; model preprocessing uses the validated one-million-pixel budget. The pinned
official layout prompt, temperature 0, top-p 1, 8,192-token server context,
6,144-output-token cap, and 300-second per-page request timeout are recorded.
The output cap is higher than the small two-page development run's 4,096 tokens,
allowing more room for dense tables within the same server context.

A failed page is retained with its complete available response, diagnostic HTML,
logs, and source; the next page is still attempted. A dead GPU server is restarted
before a later request, without silently retrying the failed page or changing its
inference settings. SIGTERM interrupts the controller and releases its GPU server.
A 300-second socket timeout is not a guarantee of immediate server cancellation.

## Outputs and interpretation

Each case contains original per-page jobs under `page-jobs/`, plus aggregate
JSON, Markdown, source crops, HTML review, status, and an inference manifest.
Aggregate envelopes explicitly map each single-page job back to its parent page;
response content and image hashes are retained. Canonical output includes only
pages whose inference and normalization completed. The final HTML identifies
partial results as finished with errors rather than claiming failed pages are
still processing. Accepted pages are not necessarily accurate pages.

The NASA PaddleOCR reference reuses its preserved first 20 raw predictions after
all selected source pixels match. No new PaddleOCR inference is performed. Its
original per-page timings and original full-run GPU memory scope are retained;
original wall time is not fabricated for the derivative subset. The diary's
existing five-page PaddleOCR result is used directly. Comparison reports cover
accepted, matching pages only and measure agreement, not reference accuracy.
Failures remain in the batch totals and are not hidden by these comparisons.

At case completion, the script creates model-named, checksum-inventoried snapshots
and releases the GPU after both cases. Existing outputs and frozen baselines remain
intact. Page jobs and reports are local, ignored by Git, and excluded from builds.

## Measured results (2026-10-30 UTC)

| Case | Attempted | Accepted | Failed | Sum of page attempt wall times |
| --- | ---: | ---: | ---: | ---: |
| NASA first 20 | 20 | 20 | 0 | 544.1 s |
| Diary selection | 5 | 3 | 2 | 289.1 s |

These sums include per-page preparation and validation, exclude shared server
startup and final exports, and include failed diary attempts. Sampled GPU-device
memory peaked at 5,914 MiB; this is not an allocator peak measurement.

NASA yielded six table elements. An assistant source-image spot check on physical
page 17 verified 24/24 selected budget values for both parsers. This is a narrow
numeric check, not independent human scoring or full table-structure accuracy:
the parsers represented merged table columns differently. The check and source
coordinates are preserved in `numeric-budget-spot-checks.json` in the NASA run.

Diary source pages 5 and 7 exhausted the 6,144-token output budget and were
rejected as incomplete generations. Raw responses and individual diagnostic HTML
reports remain saved. Accepted source pages are 4, 6 and 74. Page 4 extracted
printed title text but missed faint handwriting; page 6 produced two printed
tables; page 74 produced a handwriting transcription that still needs independent
source-based correction. Schema acceptance does not establish transcription
accuracy. The mixed-content selection includes printed reference tables and is
not a five-page pure-handwriting benchmark.

Both cases have JSON, Markdown, PDF/output HTML, source-page mappings, raw
responses and model-named immutable snapshots. The diary snapshot is explicitly
`completed_with_errors`, with three accepted pages out of five. The GPU server
was stopped after the batch. No failed page was silently retried.
