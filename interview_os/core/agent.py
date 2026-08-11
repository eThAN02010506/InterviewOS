"""Agent base class - the core abstraction of our self-built runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

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

    async def think(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": self._system_prompt}]
        history = self.memory.short_term.to_llm_messages(n=5)
        messages.extend(history)
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        if self.llm_client is None:
            return "[LLM not configured - returning placeholder]"

        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
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
        """Generate validated JSON, retrying once on invalid output.

        Local models occasionally drift from the JSON schema. The provider gets
        the schema out-of-band when supported, and an invalid first response is
        retried once with a compact, privacy-safe description of the exact
        validation failure. On a second invalid output we raise so the caller
        can fall back.
        """
        last_error: Exception | None = None
        schema = model.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        required_fields = ", ".join(str(item) for item in schema.get("required", []))
        repair_reason = ""
        for attempt in (1, 2):
            attempt_prompt = prompt
            if repair_reason:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response failed JSON validation. Correct the response using "
                    f"this diagnostic: {repair_reason}. "
                    f"Required fields: {required_fields or 'follow the supplied schema'}. "
                    "Return only the corrected JSON object; do not add prose or Markdown."
                )
            raw = await self.think(
                attempt_prompt,
                context=context,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format=response_format,
            )
            parsed, repair_reason = self._parse_structured_with_reason(raw, model)
            if parsed is not None:
                return parsed
            last_error = ValueError(f"Invalid {model.__name__} output: {repair_reason}")
            if self.debug_events is not None:
                retrying = attempt == 1
                self.debug_events.record(
                    DebugEvent(
                        level=DebugLevel.WARNING,
                        category="model",
                        action="structured_output_retry"
                        if retrying
                        else "structured_output_fallback",
                        session_id=self.session_id,
                        agent=self.name,
                        detail=(
                            f"{model.__name__} validation failed: {repair_reason}; "
                            + (
                                "retrying with targeted repair"
                                if retrying
                                else "using caller fallback"
                            )
                        ),
                        metadata={
                            "attempt": attempt,
                            "response_chars": len(raw),
                            "validation_reason": repair_reason,
                            "structured_mode": "json_schema",
                        },
                    )
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unreachable: {model.__name__} parsing failed")

    def _parse_structured(self, raw: str, model: type[StructuredModelT]) -> StructuredModelT | None:
        """Parse raw LLM output once without raising; returns None on failure.

        Used by optional generation passes (e.g. per-question answer frameworks)
        that have a deterministic fallback and should not pay a retry cost.
        """
        try:
            return parse_model_output(raw, model)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_structured_with_reason(
        raw: str, model: type[StructuredModelT]
    ) -> tuple[StructuredModelT | None, str]:
        """Validate output and return a diagnostic that contains no model text."""
        try:
            return parse_model_output(raw, model), ""
        except ValidationError as exc:
            issues = []
            for item in exc.errors(include_url=False, include_context=False, include_input=False)[
                :8
            ]:
                location = ".".join(str(part) for part in item.get("loc", ())) or "root"
                issues.append(f"{location}:{item.get('type', 'invalid_value')}")
            return None, "schema_validation[" + ",".join(issues) + "]"
        except TypeError:
            return None, "root_type:not_json_object"
        except ValueError:
            return None, "json_syntax:no_valid_object"

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
