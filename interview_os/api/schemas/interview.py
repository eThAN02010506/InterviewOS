"""Pydantic schemas for interview API."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from interview_os.core.state import ResumeClaimStatus


class StartSessionRequest(BaseModel):
    candidate_name: str = ""
    job_title: str = ""
    company_name: str = ""


class InterviewSessionResponse(BaseModel):
    id: str
    candidate_name: str = ""
    job_title: str = ""
    status: str = ""
    state: dict[str, Any] | None = None


class ResumeAnalysisRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1)


class JobAnalysisRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1)


class JobRequirementUpdateRequest(BaseModel):
    action: Literal["confirm", "edit", "delete"]
    text: str = Field(default="", max_length=1000)


class CompanyAnalysisRequest(BaseModel):
    session_id: str
    name: str = Field(min_length=1)
    context: str = ""


class InterviewerAnalysisRequest(BaseModel):
    session_id: str
    name: str = Field(min_length=1)
    position: str = ""
    company: str = ""
    public_info: str = ""


class AnalysisResponse(BaseModel):
    session_id: str
    agent: str
    output: str
    state: dict[str, Any]


class CandidatePrepRequest(BaseModel):
    session_id: str
    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    company_context: str = ""
    interviewer_name: str = ""
    interviewer_position: str = ""
    interviewer_public_info: str = ""


class EnterpriseDesignRequest(BaseModel):
    session_id: str
    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    company_context: str = ""


class AutopilotRequest(BaseModel):
    role: str = Field(pattern="^(candidate|interviewer)$")
    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    company_context: str = ""
    interviewer_name: str = ""
    interviewer_position: str = ""
    authorized_public_research: bool = False


class WorkflowResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class ResumeClaimUpdateRequest(BaseModel):
    status: ResumeClaimStatus
    note: str = Field(default="", max_length=500)
    statement: str | None = Field(default=None, min_length=1, max_length=2000)


class EntityResolutionRequest(BaseModel):
    accept: bool
    proposed_name: str = Field(default="", max_length=200)


class TranscriptEntryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)
    competency: str = Field(default="综合能力", max_length=200)


class TranscriptImportRequest(BaseModel):
    entries: list[TranscriptEntryRequest] = Field(min_length=1, max_length=50)
    auto_evaluate: bool = True


class MockAnswerRequest(BaseModel):
    question_id: UUID
    answer: str = Field(min_length=1)


class MockSessionResponse(BaseModel):
    session_id: str
    mock_session: dict[str, Any]
    current_question: dict[str, Any] | None = None


class LiveInterviewStartRequest(BaseModel):
    consent_confirmed: bool


class LiveInterviewStatusRequest(BaseModel):
    status: Literal["active", "paused", "completed"]


class LiveTranscriptRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    speaker: Literal["interviewer", "candidate", "unknown"] = "unknown"


class LiveTranscriptUpdateRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=12000)
    speaker: Literal["interviewer", "candidate", "unknown"] | None = None


class LiveSuggestionDecisionRequest(BaseModel):
    status: Literal["adopted", "edited", "skipped"]
    final_question: str = Field(default="", max_length=4000)


class LiveEvidenceConfirmationRequest(BaseModel):
    question_segment_id: UUID | None = None
    question: str = Field(default="", max_length=4000)
    competency: str = Field(default="", max_length=200)


class LiveEvidenceMergeConfirmationRequest(LiveEvidenceConfirmationRequest):
    segment_ids: list[UUID] = Field(min_length=2, max_length=20)


class LiveEvidenceReevaluationRequest(BaseModel):
    question: str = Field(default="", max_length=4000)
    competency: str = Field(default="", max_length=200)


class LiveEvidenceBatchConfirmationRequest(BaseModel):
    competency: str = Field(default="", max_length=200)
