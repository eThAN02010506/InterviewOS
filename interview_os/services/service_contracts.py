"""Shared service exceptions and policy constants.

Domain service mixins import this module instead of importing the composition
facade, which keeps the dependency direction acyclic while preserving the
historical re-exports from ``interview_service``.
"""

SPEECH_FEEDBACK_PROHIBITED_TERMS = (
    "口音",
    "方言",
    "性格",
    "情绪状态",
    "焦虑",
    "抑郁",
    "健康",
    "年龄",
    "性别",
    "族裔",
    "种族",
    "accent",
    "personality",
    "mental health",
    "ethnicity",
    "race",
    "gender",
    "age",
)


class SessionNotFoundError(LookupError):
    """The requested owner-scoped interview session does not exist."""


class WorkflowExecutionError(RuntimeError):
    """A persisted multi-Agent workflow did not complete successfully."""


class MockInterviewStateError(RuntimeError):
    """A mock-interview operation is invalid for the current state."""


class EvaluationStateError(RuntimeError):
    """The available evidence cannot currently produce an evaluation."""


class ResumeReviewStateError(RuntimeError):
    """A resume or JD review transition is invalid."""


class CandidateSessionStateError(RuntimeError):
    """Candidate-derived state cannot safely be replaced in this session."""


class LiveInterviewStateError(RuntimeError):
    """A live-interview operation is invalid for the current state."""
