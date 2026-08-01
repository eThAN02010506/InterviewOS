"""State management for the interview lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator

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


class AutopilotStatus(str, Enum):
    OFF = "off"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"


class AutopilotState(BaseModel):
    enabled: bool = False
    status: AutopilotStatus = AutopilotStatus.OFF
    phase: str = ""
    completed_actions: list[str] = Field(default_factory=list)
    pause_reason: str = ""
    authorized_public_research: bool = False
    started_at: datetime | None = None
    updated_at: datetime | None = None


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


class ResumeIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResumeClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    CONFIRMED = "confirmed"
    NEEDS_DOCUMENTS = "needs_documents"
    DISPUTED = "disputed"


class ResumeFileMetadata(BaseModel):
    filename: str = ""
    file_type: str = ""
    size_bytes: int = 0
    page_count: int = 0
    character_count: int = 0


class ResumeValidationIssue(BaseModel):
    code: str
    severity: ResumeIssueSeverity
    message: str
    field: str = ""


class ResumeClaim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: str
    statement: str
    status: ResumeClaimStatus = ResumeClaimStatus.UNVERIFIED
    verification_method: str = "candidate_confirmation"
    note: str = ""


class ResumeReview(BaseModel):
    metadata: ResumeFileMetadata = Field(default_factory=ResumeFileMetadata)
    issues: list[ResumeValidationIssue] = Field(default_factory=list)
    claims: list[ResumeClaim] = Field(default_factory=list)
    reviewed_at: datetime | None = None


class JobDescription(BaseModel):
    title: str = ""
    raw_description: str = ""
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
    public_research_status: str = "not_requested"


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
    public_research_status: str = "not_requested"


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
    content: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("content_score", "content"),
        serialization_alias="content",
    )
    technical_depth: float = Field(ge=0.0, le=1.0)
    structure: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    improved_answer: str = ""
    observed_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)

    @field_validator("content", "technical_depth", "structure", "impact", mode="before")
    @classmethod
    def normalize_percentage_score(cls, value: Any) -> Any:
        """Accept the common local-model 0-10/0-100 score convention."""
        if isinstance(value, (int, float)) and 1 < value <= 10:
            return value / 10
        if isinstance(value, (int, float)) and 10 < value <= 100:
            return value / 100
        return value

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


class HiringRecommendation(str, Enum):
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    LEAN_HIRE = "lean_hire"
    LEAN_NO_HIRE = "lean_no_hire"
    NO_HIRE = "no_hire"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CompetencyEvaluation(BaseModel):
    competency: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    competencies: list[CompetencyEvaluation] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: HiringRecommendation = HiringRecommendation.INSUFFICIENT_EVIDENCE
    summary: str = ""
    risks: list[str] = Field(default_factory=list)
    finalized_at: datetime | None = None


class FeedbackReport(BaseModel):
    overall: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)
    interviewer_notes: list[str] = Field(default_factory=list)
    recommendation_reasoning: str = ""


class InterviewState(BaseModel):
    candidate: CandidateProfile = Field(default_factory=CandidateProfile)
    resume_review: ResumeReview = Field(default_factory=ResumeReview)
    job: JobDescription = Field(default_factory=JobDescription)
    company: CompanyInfo = Field(default_factory=CompanyInfo)
    interviewer: InterviewerProfile = Field(default_factory=InterviewerProfile)
    strategy: InterviewStrategy = Field(default_factory=InterviewStrategy)
    blueprint: InterviewBlueprint = Field(default_factory=InterviewBlueprint)
    mock_interview: MockInterviewPlan = Field(default_factory=MockInterviewPlan)
    mock_session: MockInterviewSession = Field(default_factory=MockInterviewSession)
    evaluation: EvaluationReport = Field(default_factory=EvaluationReport)
    feedback: FeedbackReport = Field(default_factory=FeedbackReport)
    workflow: WorkflowProgress = Field(default_factory=WorkflowProgress)
    autopilot: AutopilotState = Field(default_factory=AutopilotState)
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
