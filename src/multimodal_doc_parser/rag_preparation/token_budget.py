"""Explicit tokenizer and lossless Unicode-safe bounded splitting."""

import os
import re
from pathlib import Path
from typing import Any


class TokenBudget:
    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ValueError("Install RAG support: uv sync --extra rag") from exc
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tokenizers").resolve()))
        self.encoding: Any = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def spans(self, text: str, limit: int) -> list[tuple[int, int]]:
        """Keep every character; prefer paragraph/sentence boundaries, then words."""
        if limit < 1:
            raise ValueError("A positive token budget is required")
        spans = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if self.count(text[start:middle]) <= limit:
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("Budget cannot hold a single Unicode character")
            if end < len(text):
                candidate = text[start:end]
                boundaries = list(re.finditer(r"\n\s*\n|(?<=[.!?])\s+", candidate))
                if not boundaries:
                    boundaries = list(re.finditer(r"\s+", candidate))
                eligible = [m.end() for m in boundaries if m.end() >= len(candidate) // 2]
                if eligible:
                    end = start + eligible[-1]
            spans.append((start, end))
            start = end
        return spans
