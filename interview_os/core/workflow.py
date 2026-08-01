"""Declarative workflow topology.

Concrete input values belong to the application service; this module only defines
the stable agent ordering and deliberately performs no I/O.
"""
from __future__ import annotations

from typing import ClassVar


class InterviewWorkflow:
    ENTERPRISE_DESIGN_AGENTS: ClassVar[tuple[str, ...]] = (
        "candidate_agent",
        "job_agent",
        "company_agent",
        "interview_design_agent",
    )
    CANDIDATE_PREP_AGENTS: ClassVar[tuple[str, ...]] = (
        "candidate_agent",
        "job_agent",
        "company_agent",
        "interviewer_agent",
        "interview_strategy_agent",
        "mock_interview_agent",
    )
    EVALUATION_AGENTS: ClassVar[tuple[str, ...]] = (
        "evaluation_agent",
        "feedback_agent",
    )
