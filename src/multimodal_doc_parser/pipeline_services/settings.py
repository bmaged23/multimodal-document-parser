"""Environment-owned deployment settings; secrets never enter job requests."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from multimodal_doc_parser.visual_enrichment.qwen_cloud import configuration


@dataclass
class ServiceSettings:
    storage: Path
    api_key: str = field(repr=False)
    urls: dict[str, str] = field(default_factory=dict)
    paddle_runtime: Path = Path("backends/paddleocr_runtime")
    dots_url: str = "http://127.0.0.1:8010/v1"
    dots_binary: Path | None = None
    dots_models: Path = Path(".cache/dots_mocr")
    dots_startup_seconds: float = 180
    embedding_cache: Path = Path(".cache/embeddings")

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        values = configuration(names={"PIPELINE_API_KEY", "OLLAMA_API_KEY"})
        for name in ["PIPELINE_API_KEY", "OLLAMA_API_KEY"]:
            secret_file = os.getenv(name + "_FILE")
            if secret_file:
                values[name] = Path(secret_file).read_text().strip()
        key = values.get("PIPELINE_API_KEY", "")
        if len(key) < 24:
            raise ValueError("Configure a PIPELINE_API_KEY of at least 24 characters")
        if values.get("OLLAMA_API_KEY"):
            os.environ["OLLAMA_API_KEY"] = values["OLLAMA_API_KEY"]
        ports = dict(documents=8101, parsing=8102, chunking=8103, indexing=8104, retrieval=8105)
        return cls(
            storage=Path(os.getenv("PIPELINE_STORAGE", "outputs/service_data")),
            api_key=key,
            urls={
                stage: os.getenv(stage.upper() + "_SERVICE_URL", f"http://127.0.0.1:{port}")
                for stage, port in ports.items()
            },
            paddle_runtime=Path(
                os.getenv("PADDLE_RUNTIME_DIRECTORY", "backends/paddleocr_runtime")
            ),
            dots_url=os.getenv("DOTS_SERVICE_URL", "http://127.0.0.1:8010/v1"),
            dots_binary=Path(os.environ["DOTS_SERVER_BINARY"])
            if os.getenv("DOTS_SERVER_BINARY")
            else None,
            dots_models=Path(os.getenv("DOTS_MODEL_DIRECTORY", ".cache/dots_mocr")),
            embedding_cache=Path(os.getenv("EMBEDDING_CACHE", ".cache/embeddings")),
        )
