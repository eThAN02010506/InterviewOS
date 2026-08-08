"""Tests for the Agent Runtime core."""
import pytest

from interview_os.core.evidence import Evidence
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
