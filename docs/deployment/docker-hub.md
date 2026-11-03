# Docker deployment guide

The complete guide moved to the [Docker README](../docker/README.md), including
cloning, pulling the public image, secret setup, Compose, GPU requirements and
release verification. Automated publishing is documented in the [CI/CD README](../ci-cd/README.md).

## Current dots release

The production image now selects quantized dots Q4_K_M + Q8_0. Earlier Paddle
release records above are historical. Use the [current Docker guide](../docker/README.md)
for tag 0.1.0-rc.4, GPU constraints, managed serving and first-use model caching.
