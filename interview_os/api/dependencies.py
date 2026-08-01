"""FastAPI dependency providers."""
from fastapi import Request

from interview_os.services.interview_service import InterviewService


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service
