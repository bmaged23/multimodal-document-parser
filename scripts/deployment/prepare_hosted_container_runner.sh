#!/usr/bin/env bash
# Only ephemeral GitHub-hosted Linux runners; never a workstation/self-hosted runner.
set -euo pipefail
if [[ "${GITHUB_ACTIONS:-}" != "true" || "${RUNNER_ENVIRONMENT:-}" != "github-hosted" || "${RUNNER_OS:-}" != "Linux" ]]; then
  echo 'Refusing cleanup outside an ephemeral GitHub-hosted Linux runner.' >&2
  exit 1
fi
# This Python/Docker job does not use the runner's Android/.NET/Swift/CodeQL SDKs.
sudo rm -rf -- /usr/local/lib/android /usr/share/dotnet /opt/ghc \
  /usr/local/share/powershell /opt/hostedtoolcache/CodeQL \
  /usr/local/swift /usr/share/swift
python3 - <<'PY'
import shutil
from pathlib import Path
root = Path('/var/lib/docker')
free = shutil.disk_usage(root if root.exists() else Path('/')).free
minimum = 40 * 1024**3
print(f'Free Docker build storage: {free / 1024**3:.1f} GiB; required: 40 GiB')
if free < minimum:
    raise SystemExit('Insufficient space for GPU layers/export/unpack; use a larger runner.')
PY
