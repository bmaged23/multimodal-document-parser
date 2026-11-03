# Unlimited-OCR 2: saved GPU findings

The third evaluated model is `baidu/Unlimited-OCR-2`, pinned to revision
`c14071b549f4fcab10909f3c4acc6a2edb366947`. Runs used an **NVIDIA GeForce GTX
1660 Ti, 6 GB VRAM (6,144 MiB)** under WSL2, with NF4 language weights, FP16
vision/projector/output head and FP16 compute. Upstream compatibility changes
include the PyTorch attention fallback, sparse routing retained, and casting
BF16 image tensors to FP16. This is an experimental configuration, not official
full-precision quality or an isolated measurement of quantization effects.

| Saved case | Completed generation | Failure/limit | Timing scope |
| --- | --- | --- | --- |
| NASA first 20 selected pages | 20/20 | None recorded | Median 108.1 s generation-only; repeated per-page model loading/startup excluded |
| Diary selection, original pages 4, 5, 7, 6, 74 | 3/5 | Original PDF pages 4 and 7 exceeded the 720 s per-job limit including loading | Successful generations only are timed; failures remain in completion totals |
| Original private two-page development sample | 2/2 | Selected table cell assertions 20/20; some labels/order still wrong | A small assistant-reviewed regression case, not an independent benchmark |

Generation completion is not extraction accuracy. This runner preserves raw
`<BLOCK>...<FORK>...` responses and diagnostic Markdown/HTML. A later, separate
[demo normalizer](../demo/README.md) converts retained native blocks for comparison;
Unlimited remains outside the production CLI/API parser choices. Its timings and Torch allocator memory
are not directly comparable to sampled total-device memory or another backend's
initialization/preprocessing scope. NASA's 20-page selection has no independent
transcription reference. No universal quality winner follows from these runs.

The [three-model profile](../../evaluation/results/parser-comparison-showcase-v1.json)
records the measured NASA observations. Raw responses, failure statuses, source
page maps and inventories remain in the model-named local archive referenced by
[evaluation/preserved-runs.json](../../evaluation/preserved-runs.json).

[Runtime profile and original findings](../../backends/unlimited_ocr_2_runtime/README.md) ·
[Interactive public excerpts](https://bmaged23.github.io/multimodal-document-parser/) ·
[Complete evaluation guide](../parsing-evaluation/README.md)
