"""BGE-large ONNX embeddings with explicit CUDA and input-window verification."""

import hashlib
import importlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_REPOSITORY = "Qdrant/bge-large-en-v1.5-onnx"
DIMENSIONS = 1024
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class GPUEmbeddings:
    """Use normalized CLS pooling; never silently truncate or use a CPU model."""

    def __init__(
        self, cache: Path = Path(".cache/embeddings"), revision: str | None = None
    ) -> None:
        try:
            ort = importlib.import_module("onnxruntime")
            hub = importlib.import_module("huggingface_hub")
            Tokenizer = importlib.import_module("tokenizers").Tokenizer
        except ImportError as exc:
            raise ValueError("Install GPU RAG support: uv sync --extra rag") from exc
        ort.preload_dlls(directory="")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ValueError("CUDA embedding provider unavailable; CPU fallback is disabled")
        cache.mkdir(parents=True, exist_ok=True)
        if revision is None:
            revision = hub.model_info(MODEL_REPOSITORY).sha
        if not revision:
            raise ValueError("Could not resolve embedding model revision")
        self.revision = revision
        self.directory = Path(
            hub.snapshot_download(
                MODEL_REPOSITORY,
                revision=revision,
                cache_dir=str(cache / "models"),
                allow_patterns=["*.json", "model.onnx", "*.onnx_data"],
            )
        )
        self.tokenizer: Any = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(cache / "cuda-embedding-profile")
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        self.session: Any = ort.InferenceSession(
            str(self.directory / "model.onnx"),
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": 4 * 1024**3,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                )
            ],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise ValueError("CUDA session failed to initialize; CPU model fallback rejected")
        self.gpu_verified = False
        self.profile_path: str | None = None
        self.artifact_hashes: dict[str, str] | None = None

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def windows(self, text: str, maximum: int = 480) -> list[tuple[int, int]]:
        """Lossless character windows measured with BGE's own WordPiece tokenizer."""
        result = []
        start = 0
        while start < len(text):
            low, high, end = start + 1, len(text), start
            while low <= high:
                middle = (low + high) // 2
                if self.token_count(text[start:middle]) <= maximum:
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("Embedding window cannot fit one character")
            if end < len(text):
                boundary = text.rfind("\n", start + (end - start) // 2, end)
                if boundary < 0:
                    boundary = text.rfind(" ", start + (end - start) // 2, end)
                if boundary >= 0:
                    end = boundary + 1
            result.append((start, end))
            start = end
        return result

    def encode(self, texts: list[str]) -> Any:
        np = importlib.import_module("numpy")

        if any(self.token_count(text) > 512 for text in texts):
            raise ValueError("BGE input exceeds 512 tokens; split it into embedding windows")
        encoded = [self.tokenizer.encode(text) for text in texts]
        maximum = max(len(e.ids) for e in encoded)
        inputs = {
            "input_ids": np.asarray(
                [e.ids + [0] * (maximum - len(e.ids)) for e in encoded], dtype=np.int64
            ),
            "attention_mask": np.asarray(
                [[1] * len(e.ids) + [0] * (maximum - len(e.ids)) for e in encoded], dtype=np.int64
            ),
            "token_type_ids": np.zeros((len(texts), maximum), dtype=np.int64),
        }
        wanted = {i.name for i in self.session.get_inputs()}
        hidden = self.session.run(None, {k: v for k, v in inputs.items() if k in wanted})[0]
        vectors = hidden[:, 0, :] if hidden.ndim == 3 else hidden
        if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError("Expected finite 1024-dimensional BGE vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Embedding model produced zero vectors")
        if not self.gpu_verified:
            self.profile_path = self.session.end_profiling()
            events = json.loads(Path(self.profile_path).read_text())
            compute = [
                e for e in events if e.get("args", {}).get("provider") == "CUDAExecutionProvider"
            ]
            if not compute:
                raise ValueError("No CUDA node execution in ONNX profile; embedding run rejected")
            self.gpu_verified = True
        return np.asarray(vectors / norms, dtype=np.float32)

    def query(self, question: str) -> Any:
        return self.encode([QUERY_PREFIX + question])[0]

    def metadata(self) -> dict[str, Any]:
        if self.artifact_hashes is None:
            self.artifact_hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.directory.iterdir()
                if p.is_file()
            }
        return dict(
            model=MODEL_NAME,
            repository=MODEL_REPOSITORY,
            revision=self.revision,
            dimensions=DIMENSIONS,
            pooling="CLS",
            normalization="L2",
            query_prefix=QUERY_PREFIX,
            document_prefix="",
            max_input_tokens=512,
            execution_provider="CUDAExecutionProvider",
            gpu_node_execution_verified=self.gpu_verified,
            profile_path=self.profile_path,
            onnxruntime_version=version("onnxruntime-gpu"),
            artifacts=dict(self.artifact_hashes),
        )
