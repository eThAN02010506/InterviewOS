"""Agent Runtime - orchestrates multiple agents through the interview lifecycle."""
from __future__ import annotations

import logging
from typing import Any

from interview_os.core.agent import Agent
from interview_os.core.message import Message, MessageType
from interview_os.core.state import InterviewState

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Agent runtime: registers agents, routes messages, maintains global state."""

    def __init__(self, llm_client: Any = None) -> None:
        self._agents: dict[str, Agent] = {}
        self.state = InterviewState()
        self.llm_client = llm_client
        self._message_log: list[Message] = []

    def register_agent(self, agent: Agent) -> None:
        self._agents[agent.name] = agent
        logger.info("Registered agent: %s (%s)", agent.name, agent.role)

    def get_agent(self, name: str) -> Agent | None:
        return self._agents.get(name)

    @property
    def agents(self) -> dict[str, Agent]:
        return dict(self._agents)

    async def run(self, agent_name: str, instruction: str = "") -> Message:
        agent = self._agents.get(agent_name)
        if agent is None:
            return Message(
                type=MessageType.RESPONSE,
                sender="runtime",
                content=f"Agent '{agent_name}' not found",
            )
        logger.info("Runtime -> %s: %s", agent_name, instruction[:80])
        result = await agent.execute(self.state, instruction)
        self._message_log.append(result)
        return result

    async def run_pipeline(self, steps: list[tuple[str, str]]) -> list[Message]:
        results = []
        for agent_name, instruction in steps:
            msg = await self.run(agent_name, instruction)
            results.append(msg)
            if msg.type == MessageType.RESPONSE:
                self.state.conversation_history.append(
                    {"agent": agent_name, "output": msg.content}
                )
        return results

    def get_message_log(self) -> list[Message]:
        return list(self._message_log)

    def reset_state(self) -> None:
        self.state = InterviewState()
        self._message_log.clear()
