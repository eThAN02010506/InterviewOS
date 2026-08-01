"""Composition root for the InterviewOS agent runtime."""
from __future__ import annotations

from typing import Any

from interview_os.agents.candidate_agent import CandidateAgent
from interview_os.agents.coach_agent import CoachAgent
from interview_os.agents.company_agent import CompanyAgent
from interview_os.agents.evaluation_agent import EvaluationAgent
from interview_os.agents.feedback_agent import FeedbackAgent
from interview_os.agents.interview_design_agent import InterviewDesignAgent
from interview_os.agents.interview_strategy_agent import InterviewStrategyAgent
from interview_os.agents.interviewer_agent import InterviewerAgent
from interview_os.agents.job_agent import JobAgent
from interview_os.agents.live_interview_agent import LiveInterviewAgent
from interview_os.agents.mock_interview_agent import MockInterviewAgent
from interview_os.core.debug import DebugEventStore
from interview_os.core.runtime import AgentRuntime
from interview_os.core.tool import ToolRegistry
from interview_os.tools.web_search import SearchProvider, WebSearchTool

AGENT_TYPES = (
    CandidateAgent,
    JobAgent,
    CompanyAgent,
    InterviewerAgent,
    InterviewStrategyAgent,
    InterviewDesignAgent,
    MockInterviewAgent,
    CoachAgent,
    LiveInterviewAgent,
    EvaluationAgent,
    FeedbackAgent,
)


def create_runtime(
    llm_client: Any = None,
    search_provider: SearchProvider | None = None,
    debug_events: DebugEventStore | None = None,
    session_id: str = "",
) -> AgentRuntime:
    """Build a fully registered runtime without hiding global state."""
    runtime = AgentRuntime(
        llm_client=llm_client, debug_events=debug_events, session_id=session_id
    )
    tools = ToolRegistry()
    tools.register(WebSearchTool(search_provider))
    for agent_type in AGENT_TYPES:
        runtime.register_agent(agent_type(llm_client=llm_client, tools=tools))
    return runtime
