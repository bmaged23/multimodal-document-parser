# Continuous integration and Docker Hub delivery

**Model choice:** dots.mocr is our preferred model family overall on the 70-page test corpus. Original BF16 offers the strongest overall balance; **Q4_K_M language + Q8_0 projector** is the production default for the **GTX 1660 Ti with 6 GB VRAM**. BF16 reached **8.53 GiB peak PyTorch allocation** on Colab T4. [Comparison, measured memory and limits](../evaluation/parser-model-selection.md).

The workflows validate source changes and publish a versioned GPU parsing image.
Publishing an image does not automatically deploy an internet-facing API or demo.

## Workflows and triggers

| Workflow | Trigger | Checks/output |
| --- | --- | --- |
| [Continuous integration](../../.github/workflows/continuous-integration.yml) | Push, pull request, manual, reusable workflow | Locked uv install; Ruff format/lint; strict mypy; environment check; unit/integration tests; wheel/sdist; separate Chromium browser tests |
| [Container validation](../../.github/workflows/container-build.yml) | Relevant pull requests or manual dispatch | Builds API and parsing targets; parsing runtime/API smoke test |
| [Docker Hub release](../../.github/workflows/docker-hub-release.yml) | `v*.*.*` tag or manual version input | Reusable CI gate; build/load; runtime smoke test; Docker login; version/SHA publication; SBOM/provenance; anonymous digest access |

Hosted Linux runners have no NVIDIA GPU. Their parsing checks validate package
metadata and required shared libraries, API/auth/storage, DOCX conversion, page
rendering and worker/client imports. Actual Paddle CUDA import/tensor allocation
was verified separately on the laptop GPU. No CI job reruns the evaluation corpus
or calls cloud models; those measurements remain separate evidence.

## Run the source checks locally

Start with [clone and installation](../../README.md#clone-and-install-from-source).
From the repository root:

```bash
uv sync --locked --extra api --extra chunking --group browser
uv run --locked --extra api --extra chunking ruff format --check .
uv run --locked --extra api --extra chunking ruff check .
uv run --locked --extra api --extra chunking mypy
uv run --locked document-parser-check-environment
uv run --locked --extra api --extra chunking pytest tests/unit tests/integration
uv run --locked --group browser playwright install --with-deps chromium
uv run --locked --group browser --extra chunking pytest tests/browser
uv build
```

Browser installation can require operating-system dependencies. See
[hosted CI results](https://github.com/bmaged23/multimodal-document-parser/actions).
Passing tests checks software behavior; it does not establish OCR accuracy.

## Container checks

```bash
docker build --target parsing -t document-parser:check .
uv run --locked python scripts/deployment/verify_container_image.py document-parser:check
# On an NVIDIA-enabled host, additionally verify native CUDA device discovery:
uv run --locked python scripts/deployment/verify_container_image.py document-parser:check --gpu
```

CUDA native builds need substantial temporary space on Docker's data disk. The
[runner preparation script](../../scripts/deployment/prepare_hosted_container_runner.sh)
reclaims unused toolchains only on ephemeral GitHub-hosted Linux runners and
requires 40 GiB free; it refuses local or self-hosted execution. Never run its
cleanup manually on a workstation. A failed storage check stops before the build.

## Configure publishing for a fork

Create a **public** Docker Hub repository. In GitHub, create the `docker-hub`
environment under **Settings → Environments** and configure:

| Setting | Type | Example |
| --- | --- | --- |
| `DOCKERHUB_USERNAME` | Environment variable | Your Docker Hub account |
| `DOCKERHUB_IMAGE` | Environment variable | `your-account/multimodal-document-parser` |
| `DOCKERHUB_TOKEN` | Environment secret | Docker Hub token with push permission |

Store the token only in the encrypted secret. Do not commit it, embed it in
build arguments or include it in an image. The owner repository has these
settings configured using the existing token at the owner's request. GitHub
secrets are repository/environment scoped; a connection alone does not populate
a new repository's secrets.

## Publish a release candidate

After merging and checking the intended commit, either create a new version tag
or manually dispatch the workflow. Example for a **new, unused** prerelease:

```bash
git tag -a v0.1.0-rc.4 -m 'Document parser release candidate 4'
git push origin v0.1.0-rc.4
```

Alternative, using an authenticated GitHub CLI:

```bash
gh workflow run docker-hub-release.yml \
  --repo bmaged23/multimodal-document-parser \
  --ref main -f tag=0.1.0-rc.4
```

Choose one trigger per release. Keep existing tags immutable; prereleases do not
update `latest` unless the manual `publish_latest` input is explicitly enabled. The release runs source/browser CI before image build and
smoke tests. It then publishes version/SHA tags with SPDX SBOM and SLSA provenance,
records the digest in its summary, and checks anonymous access to that digest.
Consumers should pin a version or immutable digest. The publisher also creates a
GitHub Release at the verified source tag with image metadata and setup links.
Creating a release does not rerun OCR evaluation.

## Recorded status

- [Quantized dots release](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41556010625): passed source/browser CI and image smoke tests; published `0.1.0-rc.3` and `latest`, synchronized the Docker Hub overview, and verified anonymous digest access. [Retained validation record](../../evaluation/results/container-release-validation-v2.json). Its real GPU parsing probe later failed on oversized dummy vision warmup; CPU-hosted success did not establish inference.
- [Quantized dots warmup fix](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41623880397): published `0.1.0-rc.4` and `latest`; live overview, anonymous digest and attestations verified. The exact public image passed its synthetic GPU worker probe after releasing WSL file caches; preceding host-pressure failures are retained. [Current validation record](../../evaluation/results/container-release-validation-v3.json).

- [First Python/browser CI](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41377223373): passed.
- [Initial hosted container validation](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41422591707): passed API/parsing builds and parsing runtime checks.
- Docker Hub `0.1.0-rc.1`: published locally, anonymous digest verified, SBOM/provenance present. This is not claimed as a hosted release run.
- [First automated release](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41424942053): passed and published `0.1.0-rc.2`; anonymous source-revision, SBOM and provenance checks passed.
- GitHub release environment: variables and encrypted secret configured. No live inference server deployment is claimed.
- [GitHub Release record](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41435425406): passed, with the verified image source tag.
- [Public demo deployment](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41483427046): passed for the 53-page v3 bundle, including four author-written assignment pages; live browser and PDF-hash checks passed.

- [Handwriting update CI](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41483426987): passed, including 140 tests.

[Docker instructions and evidence](../docker/README.md) ·
[Current release validation](../../evaluation/results/container-release-validation-v3.json) ·
[Historical Paddle release validation](../../evaluation/results/container-release-validation-v1.json)

## Saved-output demo deployment and package location

[Pages deployment](../../.github/workflows/publish-demo.yml) downloads the public
excerpt ZIP from the versioned GitHub Release, checks the pinned archive digest
and per-file inventory, and deploys static pages. Updating the bundle pin on
`main` or manually dispatching the workflow publishes the demo; no PDF parsing
or cloud inference occurs in that job.

The [release-record workflow](../../.github/workflows/publish-release-record.yml)
can catalog an already verified image without rebuilding it. It verifies the
public registry digest and exact source tag before creating a Release. Existing
release records and assets are preserved.

The official container package is on
[Docker Hub](https://hub.docker.com/r/bmaged23/multimodal-document-parser).
GitHub Packages is a separate registry: a Docker Hub push does not create a GHCR
package or fill GitHub's Packages sidebar. A GHCR mirror is optional, not a second
required runtime or a claim that an unpublished package exists. See the
[GitHub Container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

The parsing target builds the pinned dots CUDA native server instead of Paddle.
Hosted image checks verify shared-library linkage and quantized default contracts
without downloading model weights. Real GPU parsing remains a separate local
release check; passing CPU-hosted CI alone does not establish it.

The dots image passed [hosted container validation](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41547531105).
The release workflow also synchronizes the Docker Hub overview from
`docs/docker/docker-hub-overview.md`; publishing that Markdown in Git alone does
not update Docker Hub. The new default is quantized dots, and historical Paddle
release records above remain unchanged.
