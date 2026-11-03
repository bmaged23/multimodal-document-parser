# Multimodal Document Parser

**Turn PDF and DOCX files into source-linked JSON, Markdown and HTML review reports.**
The GPU parsing pipeline combines quantized dots.mocr (Q4_K_M language + Q8_0 projector) with optional chart/image
interpretation through Gemma on Ollama Cloud.

**Why quantized dots?** It is our default for the 6 GB laptop. Original BF16 dots
scored best on selected handwritten mathematics (32/37 vs Q4 29/37), but peaked
at **8.53 GiB PyTorch-allocated GPU memory** on Colab T4. It is not the winner
in every category; printed-text recovery is essentially tied.

[![Open the interactive PDF parsing demo](https://raw.githubusercontent.com/bmaged23/multimodal-document-parser/main/docs/assets/interactive-demo-banner.svg)](https://bmaged23.github.io/multimodal-document-parser/?v=4.1)

[Source and evaluation](https://github.com/bmaged23/multimodal-document-parser) ·
[Complete Docker setup](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/docker/README.md) ·
[API guide](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/api/README.md) ·
[Release and checksums](https://github.com/bmaged23/multimodal-document-parser/releases/tag/v0.1.0-rc.4)

## What this image does

![Animated parsing pipeline](https://raw.githubusercontent.com/bmaged23/multimodal-document-parser/main/docs/assets/parsing-pipeline.gif)

1. Prepare PDF pages, or convert DOCX to PDF using bundled LibreOffice.
2. Run quantized dots.mocr on a compatible NVIDIA GPU to extract text, layout, tables and visual regions.
3. Normalize supported output into canonical document JSON with source coordinates and visual crops.
4. Optionally send visual crops and page context to Ollama Cloud for generated descriptions and chart values.
5. Export JSON, Markdown and HTML with linked source pages, assets, provenance and visible warnings.

OCR-only processing bypasses the cloud step. Raw predictions, model replies and
failure diagnostics remain available. Descriptions and inferred structure are
not silently presented as verified source facts.

## Pull the public image

```bash
docker pull bmaged23/multimodal-document-parser:0.1.0-rc.4
```

**Platform:** Linux amd64. Live OCR requires a compatible NVIDIA GPU and host
container GPU support. Development used an NVIDIA GeForce GTX 1660 Ti with 6 GB
VRAM. The image includes its CUDA 12.6/sm75 native server; checksum-pinned Q4/Q8 weights download to a persistent cache on first use. The worker starts and releases the server per job. Paddle and BF16 weights are not bundled.

Docker Hub distributes the image; it does not host a running parsing API. The
interactive demo above displays saved outputs and needs no GPU, account or key.

## Start the APIs and workers

Clone the source to obtain the supported Compose configuration:

```bash
git clone https://github.com/bmaged23/multimodal-document-parser.git
cd multimodal-document-parser
export PARSER_IMAGE=bmaged23/multimodal-document-parser:0.1.0-rc.4
```

Create the local secret files using the **secret setup step in the
[Docker README](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/docker/README.md#clone-and-run-the-published-image)**.
It generates the service key and optionally accepts an Ollama key without printing
or replacing existing values. Then start the stack:

```bash
docker compose -f compose.parsing.yaml pull
docker compose -f compose.parsing.yaml up -d
docker compose -f compose.parsing.yaml ps
```

One image runs four independently started containers:

| Container | Responsibility |
| --- | --- |
| `documents-api` | Upload PDF/DOCX inputs and queue preparation jobs |
| `documents-worker` | Convert DOCX, render pages and publish prepared artifacts |
| `parsing-api` | Queue parsing jobs, report status and serve output artifacts |
| `parsing-worker` | GPU OCR and optional cloud visual enrichment |

APIs bind localhost by default. Only the parsing worker reserves a GPU; shared
named volumes retain document artifacts and model caches. The image runs as a
non-root user with a read-only application filesystem in Compose.

## API entry points

| Service | Address | Main routes |
| --- | --- | --- |
| Preparation | `http://127.0.0.1:8101/docs` | `POST /v1/documents`, `POST /v1/jobs`, `GET /v1/jobs/{id}` |
| Parsing | `http://127.0.0.1:8102/docs` | `POST /v1/jobs`, `GET /v1/jobs/{id}` |
| Artifact download | On the producing service | `GET /v1/artifacts/{id}` and `GET /v1/artifacts/{id}/files/{filename}` |

Use the local `PIPELINE_API_KEY` as a bearer token for data and job routes.
Long-running jobs return an ID for polling. Readiness checks confirm API/storage
availability, not that every model has already loaded. The
[API guide](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/api/README.md)
includes an upload → prepare → parse → download example.

## Optional cloud visual descriptions

Use `enrich_visuals: false` for OCR-only parsing. For chart/image descriptions,
provide an Ollama Cloud key with model access and set `enrich_visuals: true`.
The configured model is `gemma4:31b`. It executes remotely; Gemma weights and an
Ollama server are not inside this image. Visual enrichment sends document image
crops and context to the cloud provider. Credentials belong in runtime secrets,
not image layers or build arguments.

## Scope and evaluation

The repository compares PaddleOCR-VL-1.6, dots.mocr Q4/Q8, original dots.mocr BF16
and experimental Unlimited-OCR 2. This image uses **quantized dots.mocr** by default;
its native server is bundled and verified weights download separately. The optional
full chunking/RAG stack is configured separately.

PDF and `.docx` are supported; legacy `.doc` and encrypted documents are not.
Extraction quality varies with layout, handwriting, image resolution and model
limits. Source-reviewed evaluation repairs are disclosed case-specific processing,
not a guarantee that every new document is automatically corrected.

The GPU launcher disables the fixed-size dummy vision warmup that exhausted
workspace on the tested 6 GB device. Actual input pixel limits and GPU offload
remain unchanged. A synthetic text/table page completed through the real workers,
including HTML export and server teardown. This is a runtime check, not corpus accuracy.

## Release verification

The release workflow requires source/browser CI, hosted image/runtime checks and
anonymous registry verification. It publishes SPDX SBOM and SLSA build provenance.
The real GPU worker probe is a separate release check; CPU-hosted runners cannot
establish GPU inference. Earlier Paddle CUDA tests are historical and are not
credited to the new dots image. These checks establish software/runtime behavior,
not OCR accuracy. Evaluation PDFs, private outputs, credentials and model weights
are not redistributed in the container. Consult the versioned GitHub release for
the published digest and validation record.

[CI/CD process](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/ci-cd/README.md) ·
[Measured parser comparisons](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/parsing-evaluation/README.md)

## Why this profile

Original BF16 dots.mocr achieved our best selected handwritten-mathematics result
(32/37 vs quantized 29/37), with 8,739 MiB / **8.53 GiB peak PyTorch-allocated
GPU memory** on Colab T4. It does not win every category; printed-text recovery
is essentially tied. Quantized dots is the production default for the 6 GB
laptop. Peak allocation is not total device memory or a minimum VRAM requirement.
[Full comparison and memory scope](https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/evaluation/parser-model-selection.md).
