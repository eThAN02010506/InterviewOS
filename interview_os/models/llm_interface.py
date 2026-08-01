"""LLM unified interface for local models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """LLM client abstract interface.

    All agents call LLM through this interface,
    regardless of backend (Ollama / MLX / llama.cpp).
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...
