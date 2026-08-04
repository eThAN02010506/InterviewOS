"""Agent base class - the core abstraction of our self-built runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from interview_os.core.debug import DebugEvent, DebugEventStore, DebugLevel
from interview_os.core.memory import MemorySystem
from interview_os.core.message import Message, MessageType
from interview_os.core.state import InterviewState
from interview_os.core.tool import ToolRegistry
from interview_os.models.structured import parse_model_output

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


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
        self.debug_events: DebugEventStore | None = None
        self.session_id = ""

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.name}, an AI agent for InterviewOS.\n"
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"You analyze recruitment context and provide interview intelligence.\n"
            f"Always reason step by step and base your analysis on evidence."
        )

    @abstractmethod
    async def execute(self, state: InterviewState, instruction: str = "") -> Message: ...

    async def think(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        messages = [{"role": "system", "content": self._system_prompt}]
        history = self.memory.short_term.to_llm_messages(n=5)
        messages.extend(history)
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        if self.llm_client is None:
            return "[LLM not configured - returning placeholder]"

        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = await self.llm_client.chat(messages, **kwargs)
        return response

    async def think_structured(
        self,
        prompt: str,
        model: type[StructuredModelT],
        *,
        context: str = "",
        max_tokens: int | None = None,
    ) -> StructuredModelT:
        """Generate validated JSON from a single LLM call.

        On invalid output we raise so the caller can fall back deterministically;
        a repair call would re-prefill the whole context, which is the dominant
        cost on a local model.
        """
        raw = await self.think(prompt, context=context, max_tokens=max_tokens)
        try:
            return parse_model_output(raw, model)
        except (ValueError, TypeError):
            if self.debug_events is not None:
                self.debug_events.record(
                    DebugEvent(
                        level=DebugLevel.WARNING,
                        category="model",
                        action="structured_output_fallback",
                        session_id=self.session_id,
                        agent=self.name,
                        detail=f"Invalid {model.__name__} output; caller should fall back",
                    )
                )
            raise

    def make_response(self, content: str, recipient: str = "runtime") -> Message:
        return Message(
            type=MessageType.RESPONSE,
            sender=self.name,
            recipient=recipient,
            content=content,
        )

    def record_degradation(self, detail: str) -> None:
        if self.debug_events is not None:
            self.debug_events.record(
                DebugEvent(
                    level=DebugLevel.WARNING,
                    category="agent",
                    action="deterministic_fallback",
                    session_id=self.session_id,
                    agent=self.name,
                    detail=detail,
                )
            )
