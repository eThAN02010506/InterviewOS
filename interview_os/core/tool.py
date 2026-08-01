"""Tool base class and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool = True
    data: Any = None
    error: str = ""


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        ...

    def to_llm_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool must have a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        try:
            return await tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001 - registry is an isolation boundary
            return ToolResult(success=False, error=str(exc))

    def list_tools(self) -> list[dict[str, Any]]:
        return [t.to_llm_format() for t in self._tools.values()]
