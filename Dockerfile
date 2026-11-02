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

# Build the pinned CUDA runtime from source; do not ship laptop binaries or model weights.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS dots-native
ARG CUDA_ARCHITECTURES=75
ARG NATIVE_BUILD_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:2b37ffa871595f90b94164f6c510ad9a34bd7f94da21557cc9878587e163af37 \
    https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11960.tar.gz /tmp/llama.tar.gz
RUN mkdir /src && tar -xzf /tmp/llama.tar.gz -C /src --strip-components=1 \
    && ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# Link against the toolkit stub only while building; never copy it to the runtime.
RUN --mount=type=cache,id=dots-b11960-cuda126,target=/build,sharing=locked \
    cmake -S /src -B /build -G Ninja -DGGML_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES -DGGML_CUDA_USE_PDL=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
       -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_NATIVE=OFF \
       -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build --target llama-server --parallel $NATIVE_BUILD_JOBS \
    && mkdir -p /opt/dots-built && cp -a /build/bin/. /opt/dots-built/

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS dots-cuda-libraries

FROM api AS parsing
USER root
COPY --from=dots-native /opt/dots-built/ /opt/dots/bin/
COPY --from=dots-native /src/LICENSE /app/third_party/llama.cpp.LICENSE
COPY --from=dots-cuda-libraries /usr/local/cuda-12.6/ /usr/local/cuda-12.6/
COPY backends/dots_mocr_runtime/runtime-lock.json backends/dots_mocr_runtime/native-build-lock.json ./backends/dots_mocr_runtime/
ENV DOTS_SERVER_BINARY=/opt/dots/bin/llama-server DOTS_MODEL_DIRECTORY=/cache/dots_mocr \
    DOTS_SERVICE_URL=http://127.0.0.1:8010/v1 GGML_CUDA_PDL=0 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/opt/dots/bin:/usr/local/cuda-12.6/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer fonts-liberation libgomp1 \
    && rm -rf /var/lib/apt/lists/*
LABEL org.opencontainers.image.title="Multimodal Document Parser" \
    org.opencontainers.image.description="Quantized dots.mocr GPU parsing, structured exports, HTML review and Ollama Cloud visual enrichment"
USER app
