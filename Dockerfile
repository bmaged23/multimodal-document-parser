# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.24 AS uv

FROM python:3.12-slim AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PIPELINE_STORAGE=/data EMBEDDING_CACHE=/cache/embeddings \
    TIKTOKEN_CACHE_DIR=/cache/tokenizers HF_HOME=/cache/huggingface \
    HOME=/tmp UV_CACHE_DIR=/cache/uv PATH=/app/.venv/bin:$PATH
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/cache/uv \
    uv sync --locked --no-dev --no-install-project --extra api --extra chunking

# Keep the large GPU dependency layer independent of application/documentation edits.
FROM python:3.12-slim AS paddle-dependencies
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app/backends/paddleocr_runtime
COPY backends/paddleocr_runtime/pyproject.toml backends/paddleocr_runtime/uv.lock ./
RUN --mount=type=cache,target=/uv-cache \
    UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4 \
    UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1 uv sync --locked --no-dev

FROM dependencies AS api
COPY README.md THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN --mount=type=cache,target=/cache/uv \
    uv sync --locked --no-dev --extra api --extra chunking \
    && groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 --no-create-home app \
    && mkdir -p /data /cache && chown -R app:app /data /cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=3)"]
ENTRYPOINT ["/app/.venv/bin/python", "-m", "multimodal_doc_parser.pipeline_services.service_runtime"]

FROM api AS gpu
USER root
RUN --mount=type=cache,target=/cache/uv \
    uv sync --locked --no-dev --extra api --extra chunking --extra rag
USER app

FROM api AS documents
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-writer fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
USER app

# Inherit the large installed GPU environment instead of copying it into another layer.
FROM paddle-dependencies AS parsing
USER root
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PIPELINE_STORAGE=/data EMBEDDING_CACHE=/cache/embeddings \
    TIKTOKEN_CACHE_DIR=/cache/tokenizers HF_HOME=/cache/huggingface \
    HOME=/tmp UV_CACHE_DIR=/cache/uv PATH=/app/.venv/bin:$PATH \
    PADDLE_RUNTIME_DIRECTORY=/app/backends/paddleocr_runtime \
    PADDLE_PDX_CACHE_HOME=/cache/paddleocr PADDLE_HOME=/cache/paddle
COPY --from=api /app/.venv /app/.venv
COPY --from=api /app/src /app/src
COPY --from=api /app/pyproject.toml /app/uv.lock /app/README.md /app/THIRD_PARTY_NOTICES.md ./
COPY backends/paddleocr_runtime/run_inference.py ./backends/paddleocr_runtime/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer fonts-liberation libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && (getent group app >/dev/null || groupadd --gid 1000 app) \
    && (id app >/dev/null 2>&1 || useradd --uid 1000 --gid 1000 --no-create-home app) \
    && test "$(id -u app)" = 1000 \
    && mkdir -p /data /cache && chown -R app:app /data /cache
LABEL org.opencontainers.image.title="Multimodal Document Parser" \
    org.opencontainers.image.description="GPU PaddleOCR parsing, structured exports, HTML review and Ollama Cloud visual enrichment"
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=3)"]
ENTRYPOINT ["/app/.venv/bin/python", "-m", "multimodal_doc_parser.pipeline_services.service_runtime"]
