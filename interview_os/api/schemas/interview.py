"""Pydantic schemas for interview API."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


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


class WorkflowResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class MockAnswerRequest(BaseModel):
    question_id: UUID
    answer: str = Field(min_length=1)


class MockSessionResponse(BaseModel):
    session_id: str
    mock_session: dict[str, Any]
    current_question: dict[str, Any] | None = None
