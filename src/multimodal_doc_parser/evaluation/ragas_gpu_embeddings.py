"""Expose the project's existing 1024-dimensional GPU model to modern Ragas."""

import asyncio
import importlib
import threading
from typing import Any

BaseEmbedding: Any = importlib.import_module("ragas.embeddings.base").BaseRagasEmbedding


class RagasGPUEmbeddings(BaseEmbedding):  # type: ignore[misc]
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        self.lock = threading.Lock()

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        with self.lock:
            vector = self.model.encode([text])[0]
        if vector.shape != (1024,):
            raise ValueError("Ragas embeddings require 1024-dimensional GPU vectors")
        result: list[float] = vector.tolist()
        return result

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)
