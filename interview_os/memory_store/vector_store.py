"""Vector store abstraction for RAG memory."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store wrapper.

    Phase 1: in-memory placeholder.
    Phase 3: FAISS or Chroma backend.
    """

    def __init__(self, store_type: str = "faiss") -> None:
        self.store_type = store_type
        self._vectors: list[dict[str, Any]] = []

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._vectors.append({"text": text, "metadata": metadata or {}})

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        # Phase 3: replace with actual vector similarity search
        return self._vectors[:top_k]
