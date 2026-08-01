"""Evidence Tracker - recruitment decisions must be evidence-based."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    HR_ROUND = "hr_round"
    TECHNICAL_ROUND = "technical_round"
    SYSTEM_DESIGN_ROUND = "system_design_round"
    BEHAVIORAL_ROUND = "behavioral_round"
    MOCK_INTERVIEW = "mock_interview"
    RESUME_REVIEW = "resume_review"


class Evidence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    competency: str
    signal: str
    confidence: float = 0.0
    source: EvidenceSource = EvidenceSource.TECHNICAL_ROUND
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def is_strong(self) -> bool:
        return self.confidence >= 0.75

    def is_weak(self) -> bool:
        return self.confidence < 0.4
