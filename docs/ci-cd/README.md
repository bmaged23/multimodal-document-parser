# Continuous integration and Docker Hub delivery

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
# On an NVIDIA-enabled host, additionally verify real CUDA execution:
uv run --locked python scripts/deployment/verify_container_image.py document-parser:check --gpu
```

Cold GPU builds need 40–50 GB of temporary space on Docker's data disk. The
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
git tag -a v0.1.0-rc.2 -m 'Document parser release candidate 2'
git push origin v0.1.0-rc.2
```

Alternative, using an authenticated GitHub CLI:

```bash
gh workflow run docker-hub-release.yml \
  --repo bmaged23/multimodal-document-parser \
  --ref main -f tag=0.1.0-rc.2
```

Choose one trigger per release. Keep existing tags immutable; prereleases do not
update stable `latest`. The release runs source/browser CI before image build and
smoke tests. It then publishes version/SHA tags with SPDX SBOM and SLSA provenance,
records the digest in its summary, and checks anonymous access to that digest.
Consumers should pin a version or immutable digest.

## Recorded status

- [First Python/browser CI](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41377223373): passed.
- [Initial hosted container validation](https://github.com/bmaged23/multimodal-document-parser/actions/runs/41422591707): passed API/parsing builds and parsing runtime checks.
- Docker Hub `0.1.0-rc.1`: published locally, anonymous digest verified, SBOM/provenance present. This is not claimed as a hosted release run.
- GitHub release environment: variables and secret configured; the first automated publication must pass before it is described as verified.

[Docker instructions and evidence](../docker/README.md) ·
[Machine-readable release validation](../../evaluation/results/container-release-validation-v1.json)
