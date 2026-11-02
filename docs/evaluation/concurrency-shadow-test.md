# Small PaddleOCR concurrency shadow test

Status: interrupted at the user's request before the planned comparison completed.
The original NASA run was subsequently resumed with one-page calls. No concurrency
winner or speedup is established. The completed baseline trial took 364.56 seconds
for the two sample pages; average sampled GPU utilization was 91.68% and peak
sampled total device memory was 5953 MiB.

## Question and scope

Can the supported queued pipeline improve throughput by accepting two prepared
pages at once, compared with our existing one-page calls?

Both configurations use **one GPU model instance**. This is a page-window and
pipeline-overlap comparison, not a comparison of 1 versus 2 GPU model replicas,
and not an optimum Python worker-count search. The pinned native PaddleX runtime
caps local PaddleOCR VLM recognition batch size at 1 (`PADDLEOCR_VL_LOCAL_BATCH_SIZE`).
Its queued pipeline has separate input, layout, and VLM stage threads. Passing a
list of pages permits those stages to overlap across pages without loading a
second set of model weights.

The earlier GTX 1660 Ti run sampled 5959 MiB total device use out of 6144 MiB.
That is evidence of a tight memory budget, not proof that every form of concurrency
is impossible. GPU model replication and unsupported concurrent calls on one
mutable predictor are not part of this pilot.

## Protocol

- Input: unchanged NASA FY2024 financial report page images at 144 DPI.
- Pages: PDF page 29 (chart and text) and page 50 (dense financial table).
- Backend: complete PaddleOCR-VL-1.6 pipeline, one GPU model, native recognition.
- Settings: orientation classification and unwarping disabled; queues enabled.
- Warm-up: cover page, excluded from timed trials; initialization reported separately.
- Two rounds: windows 1 then 2, followed by windows 2 then 1.
- Each trial processes exactly the same two pages and saves raw results.
- Timing includes inference and result JSON export, but excludes warm-up/loading.
- Peak sampled total device memory and mean sampled GPU utilization are recorded.
- Compare ordered block labels and whitespace-normalized content against the
  first baseline trial. This checks content consistency; it does not establish
  OCR accuracy or source-coordinate equivalence.
- A provisional recommendation requires at least 5% median speedup and no content
  differences. Two rounds on two pages cannot establish a general optimum or
  statistical significance.

## Reproduction

From the repository root, after preparing the NASA document:

```bash
backends/paddleocr_runtime/.venv/bin/python \
  backends/paddleocr_runtime/benchmark_pipeline_concurrency.py \
  outputs/parsed_documents/<NASA-document-hash>-dpi-144 \
  outputs/shadow_benchmarks/<new-run-name> \
  --pages 29 50 --rounds 2
```

Results are checkpointed to `shadow-results.json`; per-round raw page outputs
are retained in the same directory. A failed trial is recorded as failed rather
than assigned a fabricated speed score. Memory is total GPU 0 usage sampled
once per second, not per-process allocator peak memory.

Implementation reference:
[PaddleOCR-VL usage and queue behavior](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html).
