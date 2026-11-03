# Docker image and deployment

[Docker Hub overview](https://hub.docker.com/r/bmaged23/multimodal-document-parser) ·
[Version-controlled overview](docker-hub-overview.md)

**Parser choice:** quantized dots.mocr Q4_K_M language + Q8_0 projector is the
production default. Original BF16 dots scored best on selected handwritten
mathematics, with **8.53 GiB peak PyTorch-allocated GPU memory** on Colab T4.
[Evaluation, memory scope and limits](../evaluation/parser-model-selection.md).

Parse PDF and `.docx` files into source-linked JSON, Markdown and HTML with a
compatible NVIDIA GPU. Optional chart/image descriptions use Gemma through the
Ollama Cloud API. Legacy `.doc` and encrypted documents are unsupported.

## Image contents

| Included | Execution |
|---|---|
| PDF rendering and LibreOffice DOCX conversion | Inside the image |
| dots.mocr native server | Checksum-pinned llama.cpp b11960, built with CUDA 12.6 |
| Quantized language model and vision projector | Downloaded on first parsing job; byte size/SHA-256 verified and cached |
| Parsing API and worker, JSON/Markdown/HTML exports | Inside the image |
| Gemma visual-description client | Included; inference and weights stay in Ollama Cloud |
| BF16 dots, Paddle Python runtime, evaluation PDFs and credentials | Not bundled |

The runtime targets **Linux amd64, CUDA compute capability 7.5** by default,
matching the GTX 1660 Ti development GPU. Rebuild with
`--build-arg CUDA_ARCHITECTURES=<your-supported-architecture>` for another GPU.
No CPU fallback is provided. The default 8,192-token context, sequential requests,
6,144-token output cap and flash-attention-off profile match the retained local setup.

## Clone and run the published image

```bash
git clone https://github.com/bmaged23/multimodal-document-parser.git
cd multimodal-document-parser
export PARSER_IMAGE=bmaged23/multimodal-document-parser:0.1.0-rc.4
docker pull "$PARSER_IMAGE"
```

No host Python, uv, CUDA toolkit, Paddle installation or separately launched dots
server is needed with this image. Install Docker/NVIDIA container GPU support
and a compatible host driver. Verify GPU visibility before submitting work.

Prepare secret files without printing credentials:

```bash
# Creates a new service key only if missing; prompts for an optional cloud key.
python3 scripts/deployment/configure_container_secrets.py
```

The [secret setup script](../../scripts/deployment/configure_container_secrets.py)
keeps credentials in the ignored `.cache/service_secrets/` directory. For OCR only,
leave the Ollama key empty and submit `enrich_visuals: false`. For generated visual
descriptions, configure a valid Ollama Cloud key and set `enrich_visuals: true`.

```bash
docker compose -f compose.parsing.yaml pull
docker compose -f compose.parsing.yaml up -d
docker compose -f compose.parsing.yaml ps
```

Preparation API: <http://127.0.0.1:8101/docs>. Parsing API:
<http://127.0.0.1:8102/docs>. Use the [upload → prepare → parse → download example](../api/README.md#upload-prepare-parse-and-download).
A parsing request may omit `backend`; it defaults to `dots-mocr`.

Four containers share persistent data and model-cache volumes. Only the parsing
worker reserves a GPU. The worker verifies CUDA discovery, downloads/validates
its two weights, starts the dots server under a GPU lease, parses sequentially,
and terminates the owned server before releasing the lease or starting cloud
visual enrichment. Errors retain logs and do not silently fall back to CPU.
API readiness does not mean weights are already downloaded or a GPU job has passed.

```bash
docker compose -f compose.parsing.yaml logs --tail 100 parsing-worker
docker compose -f compose.parsing.yaml down
```

Use `down` without `--volumes` to retain outputs and cached weights.

## Build and verify

```bash
docker build --target parsing -t document-parser:dots .
uv run python scripts/deployment/verify_container_image.py document-parser:dots
uv run python scripts/deployment/verify_container_image.py document-parser:dots --gpu
```

For the actual preparation/parsing worker check on a synthetic PDF, using already
cached weights:

```bash
uv run python scripts/deployment/verify_gpu_parsing.py document-parser:dots \
  --output-directory outputs/maintenance/gpu-release-probe-new
```

The smoke test checks non-root execution, API authentication/OpenAPI, dots default
contracts, native shared libraries, DOCX conversion and PDF rendering. `--gpu`
additionally checks native CUDA device discovery; it does not claim OCR accuracy.
Live GPU parsing requires the separate worker probe above; CPU-hosted CI cannot
establish GPU inference. Check the [current validation record](../../evaluation/results/container-release-validation-v3.json)
for its recorded status.
The build includes no model weights or credentials. First-use downloads need
internet access; subsequent jobs use the verified named model volume.

For native development, run the [quantized dots server](../../backends/dots_mocr_runtime/README.md)
separately and configure `DOTS_SERVICE_URL`. Native Paddle is an explicit
alternative; requesting `backend: paddleocr` in this dots-only image requires a
custom image with that runtime installed. The full optional RAG Compose stack
remains separate from this four-container parsing deployment.

The GPU launcher disables synthetic vision warmup, whose fixed dummy image
exceeded the tested 6 GB device workspace in `rc.3`. Actual page pixel limits,
Q4/Q8 weights and full GPU offload are unchanged. This avoids that startup failure;
other applications and large pages can still exhaust VRAM.
On Windows/WSL, also check available host RAM, commit capacity and C: free space.
Our published-image retry initially failed to allocate vision weights while Windows
reported only 652 MiB of available RAM, despite the CUDA device reporting free VRAM.
The validation record preserves these host conditions separately from OCR quality.

## Release and CI/CD

[CI/CD details](../ci-cd/README.md). Publishing requires source/browser CI,
container smoke tests and the existing Docker Hub environment credentials.
Each image is versioned; prior `0.1.0-rc.1`/`0.1.0-rc.2` Paddle images remain
historical releases and are not silently overwritten. The current dots release
uses `0.1.0-rc.4`; consult its GitHub release for the immutable image digest and
build verification record.

[Docker Hub overview](https://hub.docker.com/r/bmaged23/multimodal-document-parser) ·
[GitHub releases](https://github.com/bmaged23/multimodal-document-parser/releases)
