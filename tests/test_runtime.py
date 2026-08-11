"""Tests for the Agent Runtime core."""
import logging

import pytest

from interview_os.core.agent import Agent
from interview_os.core.evidence import Evidence
from interview_os.core.message import Message
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import InterviewStage, InterviewState


def test_evidence_strong_weak():
    strong = Evidence(competency="Python", signal="knows decorators", confidence=0.9)
    weak = Evidence(competency="Deploy", signal="never deployed", confidence=0.2)
    assert strong.is_strong()
    assert weak.is_weak()


def test_state_add_evidence():
    state = InterviewState()
    ev = Evidence(competency="System Design", signal="explained RAG", confidence=0.85)
    state.add_evidence(ev)
    assert "System Design" in state.evaluated_competencies
    assert state.evaluated_competencies["System Design"] == pytest.approx(0.85)


def test_state_mark_evaluated():
    state = InterviewState(missing_signals=["Python", "ML"])
    state.mark_evaluated("Python", 0.8)
    assert "Python" in state.evaluated_competencies
    assert "Python" not in state.missing_signals
    assert "ML" in state.missing_signals


def test_state_summary():
    state = InterviewState(current_stage=InterviewStage.TECHNICAL_DEEP_DIVE, next_action="Ask system design")
    s = state.summary()
    assert "technical_deep_dive" in s
    assert "Ask system design" in s


def test_runtime_register_and_run():
    runtime = AgentRuntime()
    assert runtime.agents == {}
    assert runtime.state.current_stage == InterviewStage.NOT_STARTED


@pytest.mark.asyncio
async def test_runtime_run_missing_agent_async():
    runtime = AgentRuntime()
    result = await runtime.run("nonexistent")
    assert "not found" in result.content


@pytest.mark.asyncio
async def test_runtime_logs_agent_input_metadata_without_sensitive_content(caplog):
    class EchoAgent(Agent):
        async def execute(self, state, instruction=""):
            return Message(sender=self.name, content="done")

    runtime = AgentRuntime()
    runtime.register_agent(
        EchoAgent(name="privacy_agent", role="Privacy test", goal="Return safely")
    )
    private_text = "PRIVATE-JLO-RESUME-CONTENT"

    with caplog.at_level(logging.INFO, logger="interview_os.core.runtime"):
        await runtime.run("privacy_agent", private_text)

    assert private_text not in caplog.text
    assert f"input_chars={len(private_text)}" in caplog.text


@pytest.mark.asyncio
async def test_local_llm_chat_stream_yields_delta_content():
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = request.content
        assert b"stream" in body
        return httpx.Response(
            200,
            text=(
                "data: {\"choices\":[{\"delta\":{\"content\":\"你\"}}]}\n\n"
                "data: {\"choices\":[{\"delta\":{\"content\":\"好\"}}]}\n\n"
                "data: [DONE]\n\n"
            ),
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    chunks = []
    async for piece in client.chat_stream([{"role": "user", "content": "hi"}]):
        chunks.append(piece)
    assert chunks == ["你", "好"]
    metrics = client.settings_status()["metrics"]
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    await client.close()


@pytest.mark.asyncio
async def test_local_gpt_oss_uses_low_reasoning_effort_to_preserve_final_content():
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    def handler(request):
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "low"
        assert payload["temperature"] == pytest.approx(0.2)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    result = await client.chat(
        [{"role": "user", "content": "Return JSON"}],
        temperature=0.2,
    )

    assert result == '{"answer":"ok"}'
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_chat_retries_transient_server_error_before_returning_content():
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="private provider body")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"score":0.8}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    result = await client.chat([{"role": "user", "content": "score"}])

    assert result == '{"score":0.8}'
    assert calls == 2
    assert client.settings_status()["metrics"]["requests"] == 2
    assert client.settings_status()["metrics"]["failures"] == 1
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_chat_returns_redacted_error_after_retry(caplog):
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    def handler(request):
        raise httpx.ConnectError("private-host-and-prompt")

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    result = await client.chat([{"role": "user", "content": "private resume"}])

    assert result == "[LLM Error: request failed]"
    assert "private-host-and-prompt" not in caplog.text
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_chat_stream_failure_raises_redacted_error():
    import httpx

    from interview_os.models.local_llm import LLMStreamError, LocalLLMClient

    def handler(request):
        raise httpx.ConnectError("boom")

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMStreamError, match="Local model stream failed"):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert client.settings_status()["metrics"]["prompt_tokens"] == 0
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_chat_stream_yields_before_response_finishes():
    import asyncio

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    gate = asyncio.Event()

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await gate.wait()
            yield b'data: {"choices":[{"delta":{"content":"last"}}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, stream=GatedStream())

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    first = await asyncio.wait_for(anext(stream), timeout=0.2)
    assert first == "first"
    assert not gate.is_set()
    gate.set()
    assert [piece async for piece in stream] == ["last"]
    await client.close()


@pytest.mark.asyncio
async def test_omni_chat_stream_yields_before_response_finishes():
    import asyncio

    import httpx

    from interview_os.models.omni_client import OmniAudioClient

    gate = asyncio.Event()

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"question"}}]}\n\n'
            await gate.wait()
            yield b'data: [DONE]\n\n'

    client = OmniAudioClient(
        base_url="http://omni.test/v1",
        model="m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=GatedStream())
        ),
    )
    stream = await client.suggest_next_question(b"audio", stream=True)
    assert not isinstance(stream, str)
    first = await asyncio.wait_for(anext(stream), timeout=0.2)
    assert first == "question"
    assert not gate.is_set()
    gate.set()
    assert [piece async for piece in stream] == []
    await client.close()
