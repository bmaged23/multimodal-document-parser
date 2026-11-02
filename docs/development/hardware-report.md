# Development hardware

Observed on 2026-10-30. These are environment observations, not model benchmarks.

| Component | Observed value |
| --- | --- |
| Operating environment | Linux x86_64 under WSL2 |
| CPU | Intel Core i7-9750H, 6 cores / 12 logical CPUs |
| RAM exposed to WSL | Initially 7.7 GiB; 11.68 GiB after approved 12 GiB cap/restart |
| Swap exposed to WSL | Initially 2 GiB; 3 GiB after the memory-limit restart |
| GPU | NVIDIA GeForce GTX 1660 Ti |
| GPU memory | 6144 MiB |
| NVIDIA driver reported by nvidia-smi | 560.76 |
| Driver-reported CUDA compatibility | 12.6 |
| Project Python | CPython 3.12.15 |
| uv | 0.12.24 |

The CUDA version printed by nvidia-smi does not establish that the CUDA toolkit
or a compatible inference framework is installed. PaddleOCR-VL-1.6 inference has now been tested on the local two-page input;
see `docs/evaluation/paddleocr-local-validation.md`. Quantized dots.mocr also ran
on the original two-page input using a locally built CUDA 12.6/sm75 runtime;
see `docs/evaluation/dots-mocr-local-validation.md`. Model-specific Python,
GPU architecture, precision, and runtime
requirements must be checked during each backend milestone.

Run backends sequentially initially. Measure memory before selecting image
resolution, batch size, quantization, or an Ollama vision model. PaddleOCR and quantized dots.mocr ran on the initial input. Dots.mocr used
Q4_K_M language weights, a Q8_0 projector, and a one-million-pixel input budget.
Its sampled device memory still reached 5,901 MiB; concurrency is not established.
Remote GPU execution may be needed for larger workloads.

The execution sandbox blocks NVML access; nvidia-smi succeeded outside it.
