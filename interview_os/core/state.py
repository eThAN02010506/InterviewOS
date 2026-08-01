"""State management for the interview lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from interview_os.core.evidence import Evidence


class InterviewStage(str, Enum):
    NOT_STARTED = "not_started"
    HR_SCREEN = "hr_screen"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"
    CULTURE_FIT = "culture_fit"
    WRAP_UP = "wrap_up"
    COMPLETED = "completed"


class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowProgress(BaseModel):
    name: str = ""
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    error: str = ""


class CandidateProfile(BaseModel):
    name: str = ""
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unique_advantages: list[str] = Field(default_factory=list)
    raw_resume_text: str = ""


class JobDescription(BaseModel):
    title: str = ""
    department: str = ""
    level: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)


class CompanyInfo(BaseModel):
    name: str = ""
    industry: str = ""
    stage: str = ""
    technology_stack: list[str] = Field(default_factory=list)
    culture: str = ""
    preferences: list[str] = Field(default_factory=list)
    dna: str = ""
    public_sources: list[dict[str, Any]] = Field(default_factory=list)


class InterviewerProfile(BaseModel):
    name: str = ""
    position: str = ""
    company: str = ""
    education: list[dict[str, Any]] = Field(default_factory=list)
    career_history: list[dict[str, Any]] = Field(default_factory=list)
    career_pattern: str = ""
    technical_focus: list[str] = Field(default_factory=list)
    communication_style: str = ""
    likely_preferences: list[str] = Field(default_factory=list)
    public_expressions: list[dict[str, Any]] = Field(default_factory=list)


class InterviewStrategy(BaseModel):
    summary: str = ""
    key_risks: list[str] = Field(default_factory=list)
    answer_framework: list[str] = Field(default_factory=list)
    topics_to_emphasize: list[str] = Field(default_factory=list)
    topics_to_avoid: list[str] = Field(default_factory=list)
    likely_questions: list[str] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question: str = ""
    competency: str = ""
    rationale: str = ""
    strong_signals: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


class InterviewRound(BaseModel):
    name: str = ""
    goal: str = ""
    questions: list[InterviewQuestion] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)


class InterviewBlueprint(BaseModel):
    position: str = ""
    rounds: list[InterviewRound] = Field(default_factory=list)


class MockInterviewPlan(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)


class AnswerEvaluation(BaseModel):
    content: float = Field(ge=0.0, le=1.0)
    technical_depth: float = Field(ge=0.0, le=1.0)
    structure: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    improved_answer: str = ""
    observed_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)

    def overall_score(self) -> float:
        return (self.content + self.technical_depth + self.structure + self.impact) / 4


class MockAnswerRecord(BaseModel):
    question_id: UUID
    question: str
    competency: str
    answer: str
    evaluation: AnswerEvaluation
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MockSessionStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"


class MockInterviewSession(BaseModel):
    status: MockSessionStatus = MockSessionStatus.IDLE
    current_question_index: int = 0
    responses: list[MockAnswerRecord] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class InterviewState(BaseModel):
    candidate: CandidateProfile = Field(default_factory=CandidateProfile)
    job: JobDescription = Field(default_factory=JobDescription)
    company: CompanyInfo = Field(default_factory=CompanyInfo)
    interviewer: InterviewerProfile = Field(default_factory=InterviewerProfile)
    strategy: InterviewStrategy = Field(default_factory=InterviewStrategy)
    blueprint: InterviewBlueprint = Field(default_factory=InterviewBlueprint)
    mock_interview: MockInterviewPlan = Field(default_factory=MockInterviewPlan)
    mock_session: MockInterviewSession = Field(default_factory=MockInterviewSession)
    workflow: WorkflowProgress = Field(default_factory=WorkflowProgress)
    current_stage: InterviewStage = InterviewStage.NOT_STARTED
    evaluated_competencies: dict[str, float] = Field(default_factory=dict)
    missing_signals: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    next_action: str = ""

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        if ev.competency in self.evaluated_competencies:
            old = self.evaluated_competencies[ev.competency]
            self.evaluated_competencies[ev.competency] = old * 0.6 + ev.confidence * 0.4
        else:
            self.evaluated_competencies[ev.competency] = ev.confidence

    def mark_evaluated(self, competency: str, score: float) -> None:
        self.evaluated_competencies[competency] = score
        if competency in self.missing_signals:
            self.missing_signals.remove(competency)

    def is_complete(self) -> bool:
        return self.current_stage == InterviewStage.COMPLETED

    def summary(self) -> str:
        return (
            f"Stage: {self.current_stage.value} | "
            f"Evaluated: {list(self.evaluated_competencies.keys())} | "
            f"Missing: {self.missing_signals} | "
            f"Next: {self.next_action}"
        )
