# Unlimited-OCR 2 GPU experiment

**Parser choice:** quantized dots.mocr (Q4_K_M + Q8_0) is the production default. Original BF16 dots achieved our best selected handwritten-mathematics result (32/37), with **8.53 GiB peak PyTorch-allocated GPU memory** on a Colab T4. [Results, memory scope and limits](../../docs/evaluation/parser-model-selection.md).

This isolated uv runtime probes `baidu/Unlimited-OCR-2`, pinned to revision
`c14071b549f4fcab10909f3c4acc6a2edb366947`. It is an experimental diagnostic
runner, not an integrated canonical parser backend. Do not claim model quality or
hardware compatibility until inspecting the actual run status and predictions.

The official model card documents Transformers BF16 inference tested on an A800
80 GB GPU. Its two weight files total 6,688,305,576 bytes, exceeding this laptop's
6 GiB VRAM before activations. This probe uses bitsandbytes NF4 language layers,
unquantized FP16 vision/projector/output head, and FP16 compute on the GTX 1660 Ti.
The pinned upstream PyTorch attention fallback replaces Triton attention kernels;
sparse routing stays enabled. A wrapper casts upstream BF16 image tensors to FP16.
These changes mean results cannot be presented as official full-precision quality
or an isolated quantization study. Original downloaded source files stay untouched.

The official Transformers example uses newer Torch and Python 3.10. This runtime
uses Python 3.12, Torch 2.7.1 CUDA 12.6 (for this driver), and Transformers 4.52.1.
Compatibility is an empirical test, not an assumption. No CPU fallback is provided.
Existing PaddleOCR and dots.mocr environments and weights are preserved.

## Reproduction

From the project root:

```bash
uv sync --project backends/unlimited_ocr_2_runtime --frozen
uv run python scripts/model_runtimes/setup_unlimited_ocr_2_runtime.py
backends/unlimited_ocr_2_runtime/.venv/bin/python \
  backends/unlimited_ocr_2_runtime/run_gpu_probe.py \
  --output-directory outputs/parser_runs/unlimited-ocr-2-new-run
uv run python scripts/visualization/build_unlimited_ocr_2_review.py \
  outputs/parser_runs/unlimited-ocr-2-new-run
```

Use a fresh destination. The probe uses the existing original two-page 144-DPI
preparation bundle; the private PDF is not distributed. Each page uses a single
base view, the upstream default prompt, greedy decoding and a 4,096-token cap.
This deliberately tests bounded single-page compatibility before long-document
inference; it does not measure the model's multi-page context capability.

Set `UNLIMITED_OCR_CACHE_DIR` to an existing model-cache directory on a larger
disk before setup and inference to reuse its pinned downloads. Both commands
must use the same value. This changes storage location, not the GPU profile.

The [four-model demo](../../docs/demo/README.md) also has a separate saved-output
normalizer for the native blocks. It preserves raw replies and table spans,
rejects invalid pages, and provides each model's own visual regions to Gemma.
This does not add Unlimited to the production CLI/API parser choices. The
original diagnostic runs and measurements below remain unchanged.

`model-lock.json` pins file sizes, official Git blob identities and LFS SHA-256
hashes. Downloads use ignored project-local storage. The supervised first run
verifies every identity before importing the reviewed code. Model output is never
evaluated as Python. In particular, the upstream image-saving helper uses `eval`
on predicted coordinates; this project does not call that helper and parses JSON
coordinates with bounds checks instead. HTML previews are sanitized.

Raw page JSON contains exact text, tokens, finish reason, page mapping, timing and
Torch allocator memory measurements. These memory metrics differ from the
sampled total-device metrics used in the older parser comparisons. Partial or
failed generations remain diagnostic output. No normalized Document JSON, table
accuracy score, or visual-description result is invented from the raw response.

The first validation supervisor waits for staged setup, then enforces a 20-minute
wall-clock limit on model loading plus inference. It produces a PDF-left/output-right
HTML report, preserves failures, and freezes a checksum inventory under
`outputs/evaluation_archive/Unlimited-OCR-2-NF4-FP16-vision_output/`. Its registry
entry uses `experimental_cases` so the canonical archive tool cannot mistake this
probe for an integrated parser. GPU allocations release when the subprocess exits.

Official sources:

- [Model card and code](https://huggingface.co/baidu/Unlimited-OCR-2)
- [Baidu repository](https://github.com/baidu/Unlimited-OCR)

## First measured result

The two-page probe completed on 2026-10-30 UTC. Loading took 24.54 s; page
inference took 139.72 s and 88.82 s (228.53 s combined). Maximum Torch allocated
memory was 3,150.14 MiB and reserved memory 3,446 MiB. These are allocator metrics,
not the sampled total-device measurements used by the other backends.

All 20 selected source-derived sensor-table cell text/origin/span checks passed.
The reference is assistant-reviewed, not independent human annotation. Chart and
flowchart crops retained the main content and both flowchart branches; the rotated
QC label was misread and checklist label/status order was split. No full-document
accuracy score is claimed. This tested single-page base views, not cross-page
context. Generation success does not establish transcription accuracy.

Actual output uses `<BLOCK>category[x0, y0, x1, y1]<FORK>content`. The diagnostic
review supports these blocks and legacy ref/det tags; image/chart coordinates are
mapped from a 1000-unit grid and validated before cropping. The raw response is
unchanged. Both initial and corrected-review snapshots are preserved separately.

## NASA and diary batch

`scripts/parser_evaluation/run_unlimited_ocr_2_evaluation_batch.py` tests the same NASA first-20
subset and final diary selection (original pages 4, 5, 7, 6, 74). It runs one GPU
process per page, sequentially, with the same NF4/FP16/base-view/4096-token profile.
Each job has a 720-second limit including model loading; failed and incomplete
jobs stay in the totals and the next page is still attempted. Model loading is
repeated to isolate hangs and CUDA failures, so total batch wall time is not a
comparison against the earlier parsers' inference-only timings. Per-page load,
inference and attempt timings are preserved separately.

The progress file is
`outputs/parser_runs/unlimited-ocr-2-nf4-evaluation-batch-01.json`. Per-case
outputs are `unlimited-ocr-2-nf4-nasa-first20-01` and
`unlimited-ocr-2-nf4-diary-five-01` under `outputs/parser_runs/`. Each case gets a
PDF-left/output-right review and an experimental model-named archive after all
its page jobs finish. Original source page mappings and old model outputs remain
preserved. No completion or accuracy claim follows merely from starting the batch.
