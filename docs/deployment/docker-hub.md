# GPU parsing image and public Docker Hub release

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
python3 scripts/verify_container_image.py multimodal-document-parsing:release-candidate --gpu
```

The smoke test checks a non-root, read-only API, writable data/cache, authentication,
OpenAPI, Paddle CUDA imports, LibreOffice and optionally a small CUDA tensor. It
makes **no model inference or cloud calls**. A passing smoke test is not an
end-to-end OCR/visual-quality benchmark.

Configure secrets with the existing local secret setup utility (see
[service deployment](../architecture/service-pipeline.md)); files live under the
ignored `.cache/service_secrets/` directory. Never pass credentials as build args.

```bash
export PARSER_IMAGE=bmaged23/multimodal-document-parser:0.1.0-rc.1
docker compose -f compose.parsing.yaml pull
docker compose -f compose.parsing.yaml up -d
```

`compose.parsing.yaml` runs independent preparation and parsing APIs/workers using
one image. APIs bind localhost ports 8101 and 8102; only the parsing worker reserves
a GPU and receives the Ollama secret. Data and model caches use named volumes.
Use the API contracts in `services/document_preparation/` and
`services/document_parsing/` to upload, prepare, parse and download artifacts.
The full secondary RAG stack remains in `compose.yaml`.

## CI/CD

- Continuous integration: locked dependencies, Ruff formatting/lint, strict mypy,
  environment checks, tests, package build and Chromium browser tests.
- Container validation: API and complete parsing targets build on relevant PRs;
  the parsing image receives the same credential-free runtime smoke test.
- Docker Hub release: version tags `v*.*.*`, or a manual release tag, first run
  reusable CI, build and smoke-test the parsing target, then publish version/SHA
  tags with SBOM and provenance. The digest is recorded in the workflow summary.
- Publishing does not automatically deploy a server. Consumers select an explicit
  version or immutable digest; prerelease tags do not update stable `latest`.

Configure the GitHub `docker-hub` environment with repository variables
`DOCKERHUB_USERNAME` and `DOCKERHUB_IMAGE` (`namespace/repository`), and secret
`DOCKERHUB_TOKEN` with write permission. Never commit the token. Create the Docker
Hub repository with **Public** visibility before pushing, then verify it can be
pulled anonymously. A successful authenticated push alone does not prove public
visibility. The namespace and authenticated account must be supplied before the
first publication; workflow files alone do not mean CI or publishing has run.

Private inputs, evaluation archives, `.env`, caches and local environments are
excluded from the Docker context. Only explicit source/runtime paths are copied.
Third-party notices accompany the application. Dependency and model licenses
remain governed by their upstream terms; model weights are not redistributed.

References: [Docker Hub visibility](https://docs.docker.com/docker-hub/repos/manage/access/),
[GitHub Actions publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
[SBOM and provenance](https://docs.docker.com/build/ci/github-actions/attestations/).

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
is in progress. Automated publishing still needs a successful workflow run before
it is described as verified.
