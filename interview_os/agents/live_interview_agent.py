"""Live Interview Agent - real-time copilot during interviews."""
from __future__ import annotations

import logging

from interview_os.core.agent import Agent
from interview_os.core.evidence import Evidence, EvidenceSource
from interview_os.core.message import Message, MessageType
from interview_os.core.state import InterviewState

logger = logging.getLogger(__name__)


class LiveInterviewAgent(Agent):
    """Real-time interview copilot.

    Processes live transcript segments and provides:
    - For interviewers: candidate signals, missing topics, suggested follow-ups
    - For candidates: what's being tested, what to remember
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="live_interview_agent",
            role="Real-time Interview Copilot",
            goal="Provide real-time suggestions during live interviews",
            **kwargs,
        )

    async def execute(self, state: InterviewState, instruction: str = "") -> Message:
        transcript = instruction
        context = (
            f"Stage: {state.current_stage.value}\n"
            f"Evaluated: {list(state.evaluated_competencies.keys())}\n"
            f"Missing: {state.missing_signals}\n"
            f"Transcript: {transcript[:500]}"
        )
        prompt = (
            "Analyze this live interview transcript segment. "
            "Identify: 1) What the interviewer is testing, "
            "2) What the candidate mentioned, "
            "3) What signals are missing, "
            "4) Suggested next question or response strategy."
        )
        raw = await self.think(prompt, context=context)

        ev = Evidence(
            competency="Live Response",
            signal=transcript[:200],
            confidence=0.6,
            source=EvidenceSource.TECHNICAL_ROUND,
        )
        state.add_evidence(ev)

        return Message(
            type=MessageType.SUGGESTION,
            sender=self.name,
            content=raw,
        )
