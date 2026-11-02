# Docker image and deployment

[Docker Hub overview](https://hub.docker.com/r/bmaged23/multimodal-document-parser) ·
[Version-controlled overview text](docker-hub-overview.md)

Parse PDF and `.docx` files with Docker and a compatible NVIDIA GPU; cloud chart/image descriptions also require an Ollama API key.
Legacy `.doc` and encrypted documents are unsupported, and extraction quality varies with the document.

This image distributes the **primary document-parsing pipeline**, not the model
comparison dataset or the complete secondary RAG stack. One image runs separate
preparation/parsing API and worker containers.

| Component | Included or executed |
| --- | --- |
| PDF rendering and DOCX conversion | Included; LibreOffice is installed |
| PaddleOCR-VL-1.6 GPU inference | Runtime included; model weights download to the cache at first use |
| Canonical JSON, Markdown and HTML review | Included |
| Chart/image descriptions | Ollama Cloud client included; Gemma executes remotely using your API key |
| dots.mocr server and quantized weights | Separate runtime; not bundled |
| Evaluation documents, private outputs and credentials | Not bundled |
| Chunking, indexing and agentic RAG deployment | Separate optional stack in `compose.yaml` |

## Clone and run the published image

Use Linux with NVIDIA container support, or Docker Desktop on Windows with its
WSL2 GPU backend. Install [Docker with Compose](https://docs.docker.com/compose/install/)
and ensure [GPU passthrough](https://docs.docker.com/compose/how-tos/gpu-support/) works.
The host also needs Git and Python 3 for the standard-library secret setup below.
The application, uv environment, Paddle runtime and LibreOffice are inside the
image; no host uv/Paddle installation is needed for this route.

```bash
git clone https://github.com/bmaged23/multimodal-document-parser.git
cd multimodal-document-parser
export PARSER_IMAGE=bmaged23/multimodal-document-parser:0.1.0-rc.2
docker pull "$PARSER_IMAGE"
```

If already cloned, start from `cd multimodal-document-parser`. Create local
secret files without printing or replacing existing values:

```bash
python3 - <<'PYTHON'
import getpass
import os
import secrets
from pathlib import Path

root = Path('.cache/service_secrets')
root.mkdir(parents=True, exist_ok=True, mode=0o700)
root.chmod(0o700)
values = {'pipeline_api_key': lambda: secrets.token_urlsafe(32),
          'ollama_api_key': lambda: getpass.getpass('Ollama key (Enter for OCR only): ')}
for name, value in values.items():
    path = root / name
    if not path.exists():
        secret = value()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(secret + '\n')
    path.chmod(0o644)
print('Local secret files ready; existing values retained.')
PYTHON

docker compose -f compose.parsing.yaml pull
docker compose -f compose.parsing.yaml up -d
docker compose -f compose.parsing.yaml ps
curl --fail http://127.0.0.1:8101/health/ready
curl --fail http://127.0.0.1:8102/health/ready
```

The secret directory is private to your host user (mode 700). Its files are
readable by the container UID through Compose mounts (mode 644), including hosts
whose user ID differs from the image's UID 1000. The private parent directory
prevents other host users from accessing those files.

Open the preparation API at <http://127.0.0.1:8101/docs> and the parsing API at
<http://127.0.0.1:8102/docs>. Follow the [upload-to-output API example](../api/README.md#upload-prepare-parse-and-download)
for real document processing. For OCR only, the Ollama secret file can be empty,
but parsing requests must set `enrich_visuals: false`. For chart/image descriptions,
provide a valid Ollama Cloud key and set `enrich_visuals: true`.

The `documents-api`, `documents-worker`, `parsing-api` and `parsing-worker`
containers share named data/model-cache volumes. Only the parsing worker reserves
a GPU; APIs bind localhost. First OCR use downloads its weights, so API readiness
does not mean the model is cached. Large-page latency depends on the document/GPU.

```bash
# Worker diagnostics and clean shutdown; named volumes are retained.
docker compose -f compose.parsing.yaml logs --tail 100 parsing-worker
docker compose -f compose.parsing.yaml down
```

Use `down` without `--volumes` to keep outputs and model caches. The optional full
RAG deployment in `compose.yaml` is a separate stack and needs additional images
and embedding dependencies; it is not provided by this parsing-only startup.

## Contents and execution

The `parsing` Docker target includes the project API/worker, PDF rendering,
LibreOffice DOCX conversion, PaddleOCR 3.6.0 / PaddlePaddle GPU 3.2.1 in an isolated
locked uv environment, canonical JSON/Markdown/HTML export, and the Ollama Cloud
client for Gemma `gemma4:31b` visual descriptions. CUDA userspace libraries come
from the locked Paddle runtime. Supported release platform: **Linux amd64**.

Allow at least 40–50 GB of temporary headroom on **Docker's data disk**, independently of
free space in the project filesystem. GPU wheels, extracted libraries, build
cache, compressed exports and local unpacked image layers coexist during a build.
About 25 GiB of free C: space proved insufficient on this laptop; deleting cache
inside Docker also does not necessarily shrink its Windows VHDX immediately. Dependency downloads are
cached and concurrency is bounded; the GPU dependency stage is independent of
application/README edits.

OCR model weights download into a persistent cache on first use. Gemma executes
on **Ollama Cloud**, not inside the image; an account with model access and a
runtime API key are required for visual enrichment. Cloud descriptions send image
crops to Ollama. The image contains integration code, not Gemma weights or a
local Ollama server. Do not describe this deployment as offline or self-contained
model inference. `enrich_visuals: false` runs OCR without cloud enrichment.

The dots.mocr comparison uses a separate external quantized GPU runtime; its
server and weights are not included in this PaddleOCR deployment image.

```bash
docker build --target parsing -t multimodal-document-parsing:release-candidate .
python3 scripts/deployment/verify_container_image.py multimodal-document-parsing:release-candidate --gpu
```

The smoke test checks a non-root, read-only API, writable data/cache, authentication,
OpenAPI, Paddle CUDA imports, LibreOffice and optionally a small CUDA tensor. It
makes **no model inference or cloud calls**. A passing smoke test is not an
end-to-end OCR/visual-quality benchmark.

Use the Compose secret-file setup above; files live under ignored
`.cache/service_secrets/`. The native API secret helper is documented in the
[API README](../api/README.md). Never pass credentials as build args.

```bash
export PARSER_IMAGE=bmaged23/multimodal-document-parser:0.1.0-rc.2
docker compose -f compose.parsing.yaml pull
docker compose -f compose.parsing.yaml up -d
```

`compose.parsing.yaml` runs independent preparation and parsing APIs/workers using
one image. APIs bind localhost ports 8101 and 8102; only the parsing worker reserves
a GPU and receives the Ollama secret. Data and model caches use named volumes.
Use the API contracts in `services/document_preparation/` and
`services/document_parsing/` to upload, prepare, parse and download artifacts.
The full secondary RAG stack remains in `compose.yaml`.

## Automated publishing

See the dedicated [CI/CD README](../ci-cd/README.md) for workflow triggers,
release configuration, checks and Docker Hub publication. The image context
excludes private inputs, evaluation archives, environments, caches and credentials.
Third-party notices accompany the application; model weights are not redistributed.

## Automated release verification

[GitHub Actions release](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41424942053)
passed source/browser CI, image build, runtime smoke tests, authenticated push
and anonymous access checks. The published version is **`0.1.0-rc.2`**.
Independent registry verification confirmed source commit
`5d3c1ba9ccf00917307e0616292632b9e232f1e3`, SPDX SBOM and SLSA provenance.

Immutable index digest: `sha256:fbd8e78741c933e04de58ab5648f55b3846cad65fc73290ba7ac8a35eb580c34`.

The automated Linux runner has no GPU. Its checks establish runtime/API behavior;
the separate CUDA tensor check below covers the earlier local `0.1.0-rc.1` image.
No release workflow reran OCR evaluation or called a cloud model.

## Local validation status (2026-11-02)

The 132-test suite, Ruff formatting/lint, strict typing, Python package build and
Compose validation passed. The completed parsing image passed the non-root API,
read-only filesystem, authentication, OpenAPI, writable storage, DOCX-to-PDF
conversion and page-rendering checks. A separate NVIDIA-enabled container imported
Paddle/PaddleOCRVL and successfully allocated a CUDA tensor on the GTX 1660 Ti.
The runtime includes `libgomp1`, required by Paddle's shared libraries.

CPU-only CI checks installed package metadata and shared-library resolution;
`libcuda.so.1` is supplied by the host NVIDIA runtime during GPU execution. Actual
CUDA imports and tensor allocation are checked with `--gpu` on a GPU host.
No container OCR inference or cloud requests were made in these smoke tests.

Docker storage was recovered in place on C: by removing unused build cache and
trimming/compacting the Docker data disk. Project data and evaluation archives
were preserved. The final local build reused its existing GPU dependency image:

```bash
docker buildx build --target parsing \
  --build-context paddle-dependencies=docker-image://multimodal-document-parsing:release-candidate \
  --load -t multimodal-document-parsing:runtime-fixed .
```

This override is a local incremental recovery path. A clean build and hosted CI
use the locked dependency stage in the Dockerfile without the override.

The public release candidate [bmaged23/multimodal-document-parser:0.1.0-rc.1](https://hub.docker.com/r/bmaged23/multimodal-document-parser)
is published. Anonymous registry access verified its index and confirmed the
Linux amd64 manifest matches the locally tested image. SPDX SBOM and SLSA build
provenance descriptors are present. No hosted CI run is claimed for this local
publication. Detailed local logs remain outside Git.

Immutable index digest: `sha256:e0db1384543d831159461cda2160a733d4003de9ef244ea6c66b0485bec95e4d`.

See [machine-readable release evidence](../../evaluation/results/container-release-validation-v1.json).

GPU CI jobs remove unused preinstalled toolchains only on ephemeral GitHub-hosted
Linux runners, then require 40 GiB free before building. The guard script refuses
workstation/self-hosted execution. The public GitHub repository is
[bmaged23/multimodal-document-parser](https://github.com/bmaged23/multimodal-document-parser).
The first [hosted CI run](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41377223373)
passed Python checks, package build and Chromium browser tests.
The `docker-hub` environment, both release variables and encrypted `DOCKERHUB_TOKEN`
are configured. The existing Docker Hub access token was reused at the owner's
request; its value is absent from source and logs. The first
[hosted container validation](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41422591707)
passed both image builds and the parsing runtime smoke test. Automated publishing
is verified by the successful release run linked above.

## GitHub Release and portable demonstration

[GitHub Release v0.1.0-rc.2](https://github.com/bmaged23/multimodal-document-parser/releases/tag/v0.1.0-rc.2)
collects the exact image record, source tag, portable public demo, file inventory
and checksums. The [live demo](https://bmaged23.github.io/multimodal-document-parser/)
serves saved excerpts without model calls. The public container package remains
on [Docker Hub](https://hub.docker.com/r/bmaged23/multimodal-document-parser);
no GHCR mirror is claimed.
