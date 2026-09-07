"""Tests for the Agent Runtime core."""
import logging

import pytest
from pydantic import BaseModel

from interview_os.core.agent import Agent
from interview_os.core.debug import DebugEventStore
from interview_os.core.evidence import Evidence
from interview_os.core.message import Message
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import InterviewStage, InterviewState


class _StructuredResult(BaseModel):
    score: float
    reason: str


class _StructuredTestAgent(Agent):
    async def execute(self, state, instruction=""):
        result = await self.think_structured(instruction, _StructuredResult)
        return Message(sender=self.name, content=result.model_dump_json())


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
async def test_llm_metrics_persistence_failure_never_breaks_chat_or_stream(
    tmp_path, monkeypatch
):
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    def handler(request):
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"streamed"}}]}\n\n'
                "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "complete"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="m",
        metrics_path=tmp_path / "llm-metrics.json",
        transport=httpx.MockTransport(handler),
    )
    assert client._metrics_store is not None

    def fail_save(payload):
        raise OSError("disk full with private path")

    monkeypatch.setattr(client._metrics_store, "save", fail_save)
    assert await client.chat([{"role": "user", "content": "hi"}]) == "complete"
    assert [piece async for piece in client.chat_stream([])] == ["streamed"]
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_reconfigure_preserves_injected_transport_and_new_base_url():
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    client = LocalLLMClient(
        base_url="http://old.test/v1",
        transport=httpx.MockTransport(handler),
    )

    await client.reconfigure(base_url="http://new.test/v1")
    result = await client.probe()

    assert result["models"] == ["local-model"]
    assert requests == ["http://new.test/v1/models"]
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_reconfigure_retires_client_after_active_request_drains():
    import asyncio

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    client = LocalLLMClient(
        base_url="http://old.test/v1",
        transport=httpx.MockTransport(handler),
    )
    old_http_client = client._client
    request_task = asyncio.create_task(
        client.chat([{"role": "user", "content": "hello"}])
    )
    await started.wait()

    await client.reconfigure(base_url="http://new.test/v1")
    await asyncio.sleep(0)

    assert not old_http_client.is_closed
    assert old_http_client in client._retired_clients
    release.set()
    assert await request_task == "ok"
    for _ in range(10):
        if old_http_client.is_closed:
            break
        await asyncio.sleep(0)
    assert old_http_client.is_closed
    await client.close()


@pytest.mark.asyncio
async def test_old_llm_generation_cannot_overwrite_new_response_format_capability():
    import asyncio
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    old_request_started = asyncio.Event()
    release_old_request = asyncio.Event()
    payloads = []

    async def handler(request):
        payload = json.loads(request.content)
        payloads.append((request.url.host, payload))
        if request.url.host == "old.test" and "response_format" in payload:
            old_request_started.set()
            await release_old_request.wait()
            return httpx.Response(400, text="unsupported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://old.test/v1",
        transport=httpx.MockTransport(handler),
    )
    old_call = asyncio.create_task(
        client.chat(
            [{"role": "user", "content": "old"}],
            response_format={"type": "json_object"},
        )
    )
    await old_request_started.wait()
    await client.reconfigure(base_url="http://new.test/v1")
    release_old_request.set()
    assert await old_call == '{"ok":true}'

    await client.chat(
        [{"role": "user", "content": "new"}],
        response_format={"type": "json_object"},
    )

    new_payload = next(payload for host, payload in payloads if host == "new.test")
    assert new_payload["response_format"] == {"type": "json_object"}
    await client.close()


@pytest.mark.asyncio
async def test_changing_llm_model_reprobes_response_format_capability():
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["model"] == "old" and "response_format" in payload:
            return httpx.Response(400, text="unsupported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        model="old",
        transport=httpx.MockTransport(handler),
    )
    await client.chat(
        [{"role": "user", "content": "old"}],
        response_format={"type": "json_object"},
    )
    await client.reconfigure(model="new")
    await client.chat(
        [{"role": "user", "content": "new"}],
        response_format={"type": "json_object"},
    )

    new_payload = next(payload for payload in payloads if payload["model"] == "new")
    assert new_payload["response_format"] == {"type": "json_object"}
    await client.close()


@pytest.mark.asyncio
async def test_old_model_request_cannot_overwrite_new_model_capability():
    import asyncio
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    old_request_started = asyncio.Event()
    release_old_request = asyncio.Event()
    payloads = []

    async def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["model"] == "old" and "response_format" in payload:
            old_request_started.set()
            await release_old_request.wait()
            return httpx.Response(400, text="unsupported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        model="old",
        transport=httpx.MockTransport(handler),
    )
    old_call = asyncio.create_task(
        client.chat(
            [{"role": "user", "content": "old"}],
            response_format={"type": "json_object"},
        )
    )
    await old_request_started.wait()
    await client.reconfigure(model="new")
    release_old_request.set()
    assert await old_call == '{"ok":true}'

    await client.chat(
        [{"role": "user", "content": "new"}],
        response_format={"type": "json_object"},
    )

    new_payload = next(payload for payload in payloads if payload["model"] == "new")
    assert new_payload["response_format"] == {"type": "json_object"}
    await client.close()


@pytest.mark.asyncio
async def test_corrupt_persistent_metrics_cannot_leak_stream_client_lease(tmp_path):
    import asyncio

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"requests":"not-a-number","failures":-3}', encoding="utf-8"
    )
    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        metrics_path=metrics_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                "data: [DONE]\n\n",
            )
        ),
    )

    assert [piece async for piece in client.chat_stream([])] == ["ok"]
    assert client.settings_status()["metrics"]["requests"] == 1
    assert client.settings_status()["metrics"]["failures"] == 0
    await asyncio.wait_for(client.close(), timeout=1)


@pytest.mark.asyncio
async def test_local_gpt_oss_uses_low_reasoning_effort_to_preserve_final_content():
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    def handler(request):
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"]["reasoning_effort"] == "low"
        assert "reasoning_effort" not in payload
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
async def test_think_structured_uses_schema_and_targeted_privacy_safe_repair():
    calls = []

    class RepairingLLM:
        async def chat(self, messages, **kwargs):
            calls.append((messages, kwargs))
            if len(calls) == 1:
                return '{"score":0.7,"private_candidate_claim":"do not log this"}'
            return '{"score":0.7,"reason":"evidence is incomplete"}'

    events = DebugEventStore()
    agent = _StructuredTestAgent(
        name="structured_test", role="test", goal="test", llm_client=RepairingLLM()
    )
    agent.debug_events = events
    result = await agent.think_structured("Score this answer", _StructuredResult)

    assert result.reason == "evidence is incomplete"
    assert len(calls) == 2
    first_kwargs = calls[0][1]
    assert first_kwargs["temperature"] == 0.0
    assert first_kwargs["response_format"]["type"] == "json_schema"
    assert first_kwargs["response_format"]["json_schema"]["strict"] is True
    repair_prompt = calls[1][0][-1]["content"]
    assert "reason:missing" in repair_prompt
    assert "private_candidate_claim" not in repair_prompt
    event = events.list_events(limit=1)[0]
    assert event.action == "structured_output_retry"
    assert event.metadata["validation_reason"] == "schema_validation[reason:missing]"
    assert "do not log this" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_local_llm_retries_without_unsupported_response_format():
    import json

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(400, text="response_format unsupported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"score":0.8}'}}]},
        )

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="local-model",
        transport=httpx.MockTransport(handler),
    )
    result = await client.chat(
        [{"role": "user", "content": "score"}],
        response_format={"type": "json_object"},
    )
    second = await client.chat(
        [{"role": "user", "content": "score again"}],
        response_format={"type": "json_object"},
    )

    assert result == '{"score":0.8}'
    assert second == '{"score":0.8}'
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    assert "response_format" not in payloads[2]
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
async def test_local_llm_chat_retries_malformed_success_before_safe_fallback():
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="gpt-oss-20b",
        transport=httpx.MockTransport(handler),
    )

    result = await client.chat([{"role": "user", "content": "private resume"}])

    assert result == "[LLM Error: invalid response]"
    assert calls == 2
    assert client.settings_status()["metrics"]["failures"] == 2
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
async def test_local_llm_chat_stream_accounts_for_partial_output_when_consumer_closes():
    import asyncio

    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    gate = asyncio.Event()

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await gate.wait()

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        api_key="local",
        model="m",
        input_cost_per_million=2,
        output_cost_per_million=4,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=GatedStream())),
    )
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    assert await anext(stream) == "first"
    await stream.aclose()

    metrics = client.settings_status()["metrics"]
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    assert metrics["estimated_cost_usd"] > 0
    await client.close()


@pytest.mark.asyncio
async def test_local_llm_ignores_non_finite_provider_token_usage():
    import httpx

    from interview_os.models.local_llm import LocalLLMClient

    client = LocalLLMClient(
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=(
                    b'{"choices":[{"message":{"content":"ok"}}],'
                    b'"usage":{"prompt_tokens":1e309,"completion_tokens":"invalid"}}'
                ),
                headers={"content-type": "application/json"},
            )
        ),
    )

    assert await client.chat([]) == "ok"
    assert client.settings_status()["metrics"]["prompt_tokens"] == 0
    assert client.settings_status()["metrics"]["completion_tokens"] == 0
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
