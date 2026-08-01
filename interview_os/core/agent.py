"""Agent base class - the core abstraction of our self-built runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from interview_os.core.memory import MemorySystem
from interview_os.core.message import Message, MessageType
from interview_os.core.state import InterviewState
from interview_os.core.tool import ToolRegistry


class Agent(ABC):
    """Agent base class.

    Each Agent has: name, role, goal, memory, tools, execute().
    Unlike generic frameworks, agents carry domain-specific interview intelligence.
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        memory: MemorySystem | None = None,
        tools: ToolRegistry | None = None,
        llm_client: Any = None,
    ) -> None:
        self.name = name
        self.role = role
        self.goal = goal
        self.memory = memory or MemorySystem()
        self.tools = tools or ToolRegistry()
        self.llm_client = llm_client
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.name}, an AI agent for InterviewOS.\n"
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"You analyze recruitment context and provide interview intelligence.\n"
            f"Always reason step by step and base your analysis on evidence."
        )

    @abstractmethod
    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        ...

    async def think(self, prompt: str, context: str = "") -> str:
        messages = [{"role": "system", "content": self._system_prompt}]
        history = self.memory.short_term.to_llm_messages(n=5)
        messages.extend(history)
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        if self.llm_client is None:
            return "[LLM not configured - returning placeholder]"

        response = await self.llm_client.chat(messages)
        return response

    def make_response(self, content: str, recipient: str = "runtime") -> Message:
        return Message(
            type=MessageType.RESPONSE,
            sender=self.name,
            recipient=recipient,
            content=content,
        )
