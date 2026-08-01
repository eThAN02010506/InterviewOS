"""Memory system: short-term conversation + long-term vector retrieval."""
from __future__ import annotations

from collections import deque
from typing import Any

from interview_os.core.message import Message


class ShortTermMemory:
    """Sliding window of recent messages."""

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_recent(self, n: int = 10) -> list[Message]:
        items = list(self._messages)
        return items[-n:] if n < len(items) else items

    def to_llm_messages(self, n: int = 10) -> list[dict[str, str]]:
        return [m.to_llm_format() for m in self.get_recent(n)]

    def clear(self) -> None:
        self._messages.clear()


class LongTermMemory:
    """Vector-based retrieval of historical context.

    Phase 1 uses in-memory dict placeholder; Phase 3 plugs in FAISS/Chroma.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def store(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._store[key] = {"content": content, "metadata": metadata or {}}

    async def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        results = []
        for key, item in self._store.items():
            if query.lower() in item["content"].lower():
                results.append({"key": key, **item})
        return results[:top_k]


class MemorySystem:
    """Full agent memory = short-term + long-term."""

    def __init__(self, max_short_term: int = 50) -> None:
        self.short_term = ShortTermMemory(max_messages=max_short_term)
        self.long_term = LongTermMemory()

    def remember(self, message: Message) -> None:
        self.short_term.add(message)

    async def recall(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return await self.long_term.retrieve(query, top_k=top_k)
