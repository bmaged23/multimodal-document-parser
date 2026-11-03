# Parser choice and GPU memory

**Production default: dots.mocr Q4_K_M language weights + Q8_0 vision projector.**
The CLI, parsing/pipeline request defaults, container worker and default demo
selection use this quantized profile. PaddleOCR remains an optional native
comparison backend; its preserved IRENA/IMF pipeline reports remain labeled Paddle.

**dots.mocr is the preferred model family overall for the retained 70-page corpus.**
Original BF16 offers the strongest balance: 91.67% visual-region recall and
80.96% label-linked row/value matching. Q4/Q8 has the lowest scoped text error
rate (0.67%); Paddle leads row/value matching (86.58%). The overall interpretation
is **BF16 → Q4/Q8 → Paddle → Unlimited**, not a weighted composite or a universal
winner claim. [Evaluation method, report and metrics](README.md).

**Why deploy Q4/Q8?** The local GTX 1660 Ti has 6 GB VRAM. Original BF16 exceeded
that capacity in the measured configuration; quantized dots provides the tested
local GPU path. The older handwriting study also favored BF16 (32/37 versus
Q4 29/37), but remains a separate selected-assertion experiment. Historical
normalization failures on the original two-page sample are preserved.

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

[Current four-PDF evaluation](README.md) ·
[Earlier 31-page study](parser-corpus-four-profiles.md) ·
[Measured memory, per-page peaks and prediction hashes](../../evaluation/results/parser-model-selection-v1.json)
