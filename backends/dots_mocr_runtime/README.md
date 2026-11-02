# dots.mocr runtime

**Parser choice:** quantized dots.mocr (Q4_K_M + Q8_0) is the production default. Original BF16 dots achieved our best selected handwritten-mathematics result (32/37), with **8.53 GiB peak PyTorch-allocated GPU memory** on a Colab T4. [Results, memory scope and limits](../../docs/evaluation/parser-model-selection.md).

The default parser uses an OpenAI-compatible HTTP server. The main uv environment
needs no Torch, vLLM, or CUDA Python packages; PaddleOCR retains its own environment.

For the 6 GiB laptop, `runtime-lock.json` pins a community GGUF conversion:
Q4_K_M language weights and a Q8_0 vision projector. Record this configuration
separately from the original BF16 model. The converter names dots.mocr as its
base but does not pin the original revision; equivalence is not verified.
The observed official source revision is metadata, not proof of conversion lineage.

```bash
uv run python scripts/model_runtimes/setup_dots_mocr_runtime.py
uv run python scripts/model_runtimes/build_dots_mocr_runtime.py --jobs 4
uv run python scripts/model_runtimes/serve_dots_mocr.py
```

Setup downloads only the quantized model and projector to `.cache/dots_mocr/`,
verifying SHA-256 and byte size. The build command obtains checksum-pinned source
and micromamba artifacts, creates the exact native toolchain from
`toolchain-linux-64.lock`, and compiles for CUDA 12.6 / sm75. The first build can
take tens of minutes on this laptop; subsequent invocations reuse build objects.
`--jobs 2` reduces concurrent compiler memory use. The main Python package still
uses uv; this separate environment contains native build tools and CUDA libraries.
It does not change the system CUDA installation.
Run the server in a separate terminal. It binds to localhost, serves one request
at a time, uses six CPU threads, and disables flash attention for the GTX 1660 Ti.
Model weights, caches, and binaries are ignored by Git and excluded from packages.

The pinned CUDA 12.8 binary initially failed on driver 560.76 with
`the provided PTX was compiled with an unsupported toolchain` during warmup.
The local source build targets CUDA 12.6 with sm75 code and disables PDL/CUDA
graphs. This configuration completed two-page GPU inference; see
[measurements and limitations](../../docs/evaluation/dots-mocr-local-validation.md).

## Parse the same prepared input

```bash
uv run document-parser parse outputs/parsed_documents/<source-hash>-dpi-144 \
  --backend dots-mocr \
  --model-version '<exact model revision and quantization label>' \
  --runtime-metadata .cache/dots_mocr/server-launch.json \
  --output-directory outputs/parser_runs/dots-mocr-local-validation
```

The default endpoint is `http://127.0.0.1:8010/v1`, with model alias
`dots.mocr-Q4_K_M`. The explicit version label should match `runtime-lock.json`.
The server's advertised model ID is checked before sending page content.
`DOTS_MOCR_API_KEY` optionally supplies authentication without writing the key to outputs.

Each request has a 300-second socket timeout, a 6,144-output-token cap by default in CLI/API (8,192-token server context), and a
1-million-pixel input budget. `--timeout-seconds`, `--max-tokens`, and `--max-pixels`
make these settings explicit. A timeout stops the client; remote server cancellation
is server-dependent. A timed-out request must not be counted as a completed page.

Pages run sequentially. Inputs resize to multiples of 28 pixels; the exact sent
PNG, its hash/dimensions, complete response, token usage, page time, runtime metadata,
and per-page checkpoints are retained under `raw/dots_mocr/`. Bounding boxes refer
to the sent image in pixels and map back to normalized source-page coordinates.
The default is the exact upstream `prompt_layout_all_en`, pinned to revision
`23f3e5612fb8066d4034d5ecfc8f33a9243533eb`. Prompt text, SHA-256, source metadata,
and MIT notice are packaged as resources. The earlier project-prompt run is
preserved separately. This integration does not reproduce upstream benchmarks.

Only `finish_reason=stop` and valid layout JSON qualify as generated pages.
Truncated JSON, unknown categories, invalid boxes, or malformed HTML tables cause
a failed run with diagnostics retained. Generated SVG/HTML from picture regions
is never executed; picture regions are source-image crops. Chart descriptions
and values remain milestone 5.

## Alternative server

A separately deployed official dots.mocr model can use `--server-kind vllm`,
`--base-url`, `--model`, and an explicit `--model-version`. This adds the image
prefix used by the upstream vLLM request format. The repository does not provision
a remote GPU or paid endpoint. Such runs must have their own model identity and
server memory measurements. The client does not claim local GPU memory for remote servers.

References: [official parser](https://github.com/studio-dots-ai/dots.mocr),
[official model](https://huggingface.co/dots-studio/dots.mocr),
[conversion](https://huggingface.co/prithivMLmods/dots.mocr-GGUF),
[llama.cpp](https://github.com/ggml-org/llama.cpp).

## Retired CPU experiment

The BF16 CPU approach was retired at the user's request after repetitive,
incomplete generation. Its active launcher/download support, probe scripts,
tests, and downloaded BF16 weights have been removed. The supported local path
is the quantized Q4_K_M language model with Q8_0 projector on GPU.

Historical diagnostics and frozen reproduction snapshots remain available; see
[the recorded experiment](../../docs/evaluation/dots-mocr-cpu-bf16-probe.md).
Reintroducing CPU inference requires a new explicit request.
