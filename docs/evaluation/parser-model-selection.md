# Parser choice and GPU memory

**Production default: dots.mocr Q4_K_M language weights + Q8_0 vision projector.**
The CLI, parsing/pipeline request defaults, container worker and default demo
selection use this quantized profile. PaddleOCR remains an optional native
comparison backend; its preserved IRENA/IMF pipeline reports remain labeled Paddle.

**Original unquantized BF16 dots.mocr achieved our best selected handwritten
mathematics result: 32/37 versus quantized dots 29/37.** It is essentially tied
with Q4 on NASA printed-text recovery and does not win every category. In
particular, its original two-page sample failed strict normalization; Paddle
finished all 31 shared pages, while BF16 finished 30 and accepted 28.

| Observed profile | GPU | Measured peak | Measurement scope |
|---|---|---:|---|
| Quantized dots Q4_K_M + Q8_0 | GTX 1660 Ti, 6,144 MiB | 5,914 MiB (5.78 GiB) | Sampled total GPU-device memory, once per second |
| Original dots BF16 | Colab Tesla T4 | 8,739.26 MiB (8.53 GiB) | Per-page maximum PyTorch allocated memory, maximum across 31 attempts |

BF16 exceeded the laptop's 6 GB VRAM in this tested profile. It needs a
higher-memory GPU for this configuration; a T4 was used for the experiment.
The 8.53 GiB figure excludes non-PyTorch allocations and is not a minimum GPU
capacity requirement. The two measurement types and hardware are different;
do not compare them as allocator-equivalent measurements or causal quantization effects.
The BF16 cloud experiment does not restore the retired BF16 CPU backend.

The production Docker image contains a CUDA 12.6 native runtime and downloads
the two checksum-pinned quantized weights to its persistent cache on first use.
BF16 weights and Paddle's Python runtime are not bundled in this release.

[All four PDFs and category-level results](parser-corpus-four-profiles.md) ·
[Measured memory, per-page peaks and prediction hashes](../../evaluation/results/parser-model-selection-v1.json)
