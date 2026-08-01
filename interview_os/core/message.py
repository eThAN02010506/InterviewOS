"""Message types for inter-agent communication."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    SUGGESTION = "suggestion"
    EVIDENCE = "evidence"


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: MessageType = MessageType.REQUEST
    role: MessageRole = MessageRole.ASSISTANT
    sender: str = ""
    recipient: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_llm_format(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}
