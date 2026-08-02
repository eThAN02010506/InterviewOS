from io import BytesIO

import httpx
from docx import Document
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage
from interview_os.models.local_llm import LocalLLMClient
from interview_os.services.settings_service import LocalSettingsStore
from interview_os.tools.asr import ASRClient
from interview_os.tools.web_search import SearchResult


class MockLLM:
    async def chat(self, messages, **kwargs):
        return '{"name":"Ada","skills":["Python"],"strengths":["Systems"]}'

    async def embed(self, text):
        return []


class WorkflowLLM:
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "structured candidate profile" in prompt:
            return '{"name":"Ada","skills":["Python"]}'
        if "job description" in prompt:
            return '{"title":"Engineer","competencies":["System Design"]}'
        if "company dna" in prompt:
            return '{"name":"Example","dna":"Engineering culture"}'
        if "fuse three inputs" in prompt:
            return '{"summary":"Show impact","key_risks":["Scope"]}'
        if "mock interview plan" in prompt:
            return '{"questions":[{"question":"Design it","competency":"System Design"}]}'
        if "analyze this interview answer" in prompt:
            return '{"content":0.8,"technical_depth":0.8,"structure":0.8,"impact":0.8,"feedback":[],"improved_answer":"Better","observed_signals":["Clear design"],"missing_signals":[]}'
        if "based on the following evidence" in prompt:
            return '{"competencies":[{"competency":"System Design","score":0.8,"confidence":0.8,"supporting_evidence":["Clear design"],"gaps":[]}],"overall_score":0.8,"recommendation":"hire","summary":"Meets the bar","risks":[]}'
        if "generate evidence-based feedback" in prompt:
            return '{"overall":"Meets the bar","strengths":["Clear design"],"improvements":[],"action_plan":["Continue practice"],"interviewer_notes":[],"recommendation_reasoning":"Evidence supports hire."}'
        if "return exactly one json object matching the questionsuggestion schema" in prompt:
            return '{"suggested_question":"你如何验证这个架构权衡？","question_type":"follow_up","competency":"System Design","rationale":"需要补充验证方法","evidence_gap":"量化验证","expected_signals":["指标","压测"],"confidence":0.8,"alternatives":["失败时如何回滚？"]}'
        raise AssertionError(prompt)

    async def embed(self, text):
        return []


class FollowupWorkflowLLM(WorkflowLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "mock interview plan" in prompt:
            return '{"questions":[{"question":"Design it","competency":"System Design","follow_ups":["What evidence supports that trade-off?"]}]}'
        if "analyze this interview answer" in prompt:
            return '{"content":0.6,"technical_depth":0.6,"structure":0.6,"impact":0.5,"feedback":["Add evidence"],"improved_answer":"Better","observed_signals":["Design thinking"],"missing_signals":["Measured impact"]}'
        return await super().chat(messages, **kwargs)


class BlueprintFallbackQuestionLLM(WorkflowLLM):
    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "interview blueprint" in prompt:
            return (
                '{"position":"Engineer","rounds":[{"name":"技术面","goal":"验证核心能力",'
                '"evaluation_criteria":["证据完整"],"questions":['
                '{"question":"请讲一次系统设计权衡。","competency":"System Design",'
                '"rationale":"核心能力","strong_signals":["权衡","指标"],"follow_ups":[]},'
                '{"question":"请讲一次事故复盘。","competency":"Incident Review",'
                '"rationale":"稳定性能力","strong_signals":["复盘","改进"],"follow_ups":[]}]}]}'
            )
        if (
            "return exactly one json object matching the questionsuggestion schema" in prompt
            or "repair the previous response into valid json only" in prompt
        ):
            return "not json"
        return await super().chat(messages, **kwargs)


class FactSearchProvider:
    async def search(self, query, limit=5, *, search_depth="basic"):
        return [
            SearchResult(
                title="Example AI platform",
                url="https://example.com/about",
                snippet="Example develops AI systems for enterprise interview workflows.",
                source="test",
                source_quality="official",
                is_official=True,
            )
        ][:limit]


def make_resume_docx() -> bytes:
    document = Document()
    document.add_paragraph("Ada ada@example.com 13800138000")
    document.add_paragraph("教育经历 Example University 本科")
    document.add_paragraph("工作经历 Example有限公司 将性能提升 30%")
    document.add_paragraph("技能 Python FastAPI")
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_session_resume_analysis_flow(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    app = create_app(storage=storage, llm_client=MockLLM(), configure_llm=False)

    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "面试智能工作台" in home.text
        assert client.get("/static/app.js").status_code == 200
        info = client.get("/", headers={"Accept": "application/json"})
        assert info.json()["name"] == "InterviewOS"

        created = client.post(
            "/api/interviews/sessions",
            json={"candidate_name": "Initial", "job_title": "Engineer"},
        )
        assert created.status_code == 200
        session_id = created.json()["id"]

        analyzed = client.post(
            "/api/analysis/resume",
            json={"session_id": session_id, "text": "Experienced Python developer"},
        )
        assert analyzed.status_code == 200
        assert analyzed.json()["state"]["candidate"]["name"] == "Ada"

        fetched = client.get(f"/api/interviews/sessions/{session_id}")
        assert fetched.status_code == 200
        assert fetched.json()["state"]["candidate"]["skills"] == ["Python"]

        debug = client.get("/api/debug/events")
        assert debug.status_code == 200
        actions = [event["action"] for event in debug.json()["events"]]
        assert "session_created" in actions
        assert "agent_completed" in actions


def test_missing_session_returns_404(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'missing.db'}")
    app = create_app(storage=storage, llm_client=MockLLM(), configure_llm=False)
    with TestClient(app) as client:
        response = client.get("/api/interviews/sessions/missing")
    assert response.status_code == 404


def test_resume_upload_and_human_confirmation_flow(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'resume-upload.db'}")
    app = create_app(storage=storage, llm_client=MockLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        uploaded = client.post(
            f"/api/resumes/{session_id}/upload",
            files={
                "file": (
                    "ada.docx",
                    make_resume_docx(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert uploaded.status_code == 200
        state = uploaded.json()["state"]
        assert "Ada" in state["candidate"]["raw_resume_text"]
        claim = state["resume_review"]["claims"][0]

        confirmed = client.patch(
            f"/api/resumes/{session_id}/claims/{claim['id']}",
            json={"status": "confirmed", "note": "候选人已确认"},
        )
        invalid = client.post(
            f"/api/resumes/{session_id}/upload",
            files={"file": ("old.doc", b"legacy", "application/msword")},
        )

    assert confirmed.status_code == 200
    assert confirmed.json()["state"]["resume_review"]["claims"][0]["status"] == "confirmed"
    assert invalid.status_code == 422


def test_settings_ui_configures_tavily_without_exposing_key(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'settings.db'}")
    llm = LocalLLMClient(base_url="http://localhost:11434/v1", api_key="local", model="test")
    app = create_app(
        storage=storage,
        llm_client=llm,
        configure_llm=False,
        settings_store=LocalSettingsStore(tmp_path / "settings.json"),
    )
    with TestClient(app) as client:
        page = client.get("/api/settings/ui")
        assert page.status_code == 200
        assert "Tavily API Key" in page.text

        updated = client.put(
            "/api/settings",
            json={"search": {"provider": "tavily", "tavily_api_key": "tvly-secret"}},
        )
        assert updated.status_code == 200
        body = updated.json()
        assert body["search"]["selected"] == "tavily"
        assert body["search"]["configured"]["tavily"] is True
    assert "tvly-secret" not in updated.text
    assert (tmp_path / "settings.json").stat().st_mode & 0o777 == 0o600


def test_candidate_prep_api_returns_structured_workflow(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'workflow-api.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        response = client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": session_id,
                "resume_text": "Python engineer",
                "job_description": "Platform engineer",
                "company_name": "Example",
            },
        )
    assert response.status_code == 200
    state = response.json()["state"]
    assert state["workflow"]["status"] == "completed"
    assert state["strategy"]["summary"] == "Show impact"
    assert state["mock_interview"]["questions"][0]["question"] == "Design it"


def test_job_requirement_review_confirms_edits_and_deletes_inferred_items(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'job-review.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        analyzed = client.post(
            "/api/analysis/job",
            json={"session_id": session_id, "text": "高级平台工程师"},
        )
        confirmed = client.patch(
            f"/api/analysis/job/{session_id}/requirements/0",
            json={"action": "confirm"},
        )
        edited = client.patch(
            f"/api/analysis/job/{session_id}/requirements/0",
            json={"action": "edit", "text": "需要设计高可用平台架构"},
        )
        deleted = client.patch(
            f"/api/analysis/job/{session_id}/requirements/0",
            json={"action": "delete"},
        )

    assert analyzed.status_code == 200
    assert analyzed.json()["state"]["job_review"]["requirements"][0]["origin"] == "inferred"
    assert confirmed.status_code == 200
    assert confirmed.json()["state"]["job_review"]["requirements"][0]["origin"] == "explicit"
    assert edited.status_code == 200
    requirement = edited.json()["state"]["job_review"]["requirements"][0]
    assert requirement == {"text": "需要设计高可用平台架构", "origin": "explicit"}
    assert deleted.status_code == 200
    assert deleted.json()["state"]["job_review"]["requirements"] == []


def test_mock_interview_api_progresses_to_completion(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'mock-api.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": session_id,
                "resume_text": "Python engineer",
                "job_description": "Platform engineer",
                "company_name": "Example",
            },
        )
        started = client.post(f"/api/mock-interviews/{session_id}/start")
        question_id = started.json()["current_question"]["id"]
        answered = client.post(
            f"/api/mock-interviews/{session_id}/answers",
            json={"question_id": question_id, "answer": "I explained the design trade-offs."},
        )
    assert answered.status_code == 200
    assert answered.json()["mock_session"]["status"] == "completed"
    assert answered.json()["current_question"] is None


def test_evaluation_api_generates_dual_side_report(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'evaluation-api.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        without_evidence = client.post(f"/api/evaluations/{session_id}")
        assert without_evidence.status_code == 409

        client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": session_id,
                "resume_text": "Python",
                "job_description": "Platform",
                "company_name": "Example",
            },
        )
        started = client.post(f"/api/mock-interviews/{session_id}/start").json()
        client.post(
            f"/api/mock-interviews/{session_id}/answers",
            json={"question_id": started["current_question"]["id"], "answer": "Clear design"},
        )
        response = client.post(f"/api/evaluations/{session_id}")
    assert response.status_code == 200
    assert response.json()["state"]["evaluation"]["recommendation"] == "insufficient_evidence"
    assert "证据不足" in response.json()["state"]["feedback"]["overall"]
    notes = response.json()["state"]["feedback"]["interviewer_notes"]
    assert notes == [
        "当前仅有 1 条证据，覆盖 1 个胜任力；未达到招聘决策门槛。",
        "证据不足不是负面证据，需要继续采集独立回答",
    ]


def test_mock_interview_asks_followup_before_advancing(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'followup.db'}")
    app = create_app(storage=storage, llm_client=FollowupWorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/workflows/candidate-prep",
            json={
                "session_id": session_id,
                "resume_text": "Python",
                "job_description": "Platform",
                "company_name": "Example",
            },
        )
        started = client.post(f"/api/mock-interviews/{session_id}/start").json()
        question_id = started["current_question"]["id"]
        first = client.post(
            f"/api/mock-interviews/{session_id}/answers",
            json={"question_id": question_id, "answer": "I made a trade-off."},
        ).json()
        assert first["current_question"]["question"] == "What evidence supports that trade-off?"
        second = client.post(
            f"/api/mock-interviews/{session_id}/answers",
            json={"question_id": question_id, "answer": "Latency fell by 30%."},
        ).json()
    assert second["mock_session"]["status"] == "completed"
    assert second["mock_session"]["responses"][1]["is_follow_up"] is True


def test_interviewer_transcript_import_generates_hiring_report(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'transcript.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        response = client.post(
            f"/api/evaluations/{session_id}/transcript",
            json={
                "entries": [
                    {
                        "competency": "System Design",
                        "question": "Design a service",
                        "answer": "I measured latency and explained the trade-offs.",
                    }
                ]
            },
        )
    assert response.status_code == 200
    result = response.json()["state"]
    assert len(result["live_interview_records"]) == 1
    assert result["evidence"][0]["source"] == "live_interview"
    assert result["evaluation"]["recommendation"] == "insufficient_evidence"
    assert "不能给出录用" in result["feedback"]["recommendation_reasoning"]


def test_live_audio_transcription_and_next_question_flow(tmp_path):
    def asr_handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(
            200,
            json={"text": "我使用压测比较两个方案，最终 P95 降低了 40%。"},
        )

    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-interview.db'}")
    asr = ASRClient(
        base_url="http://asr.test:9001",
        transport=httpx.MockTransport(asr_handler),
    )
    app = create_app(
        storage=storage,
        llm_client=WorkflowLLM(),
        configure_llm=False,
        asr_client=asr,
    )
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/analysis/job",
            json={"session_id": session_id, "text": "System Design Engineer"},
        )
        denied = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": False},
        )
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        transcribed = client.post(
            f"/api/live-interviews/{session_id}/audio",
            data={"speaker": "candidate", "language": "zh"},
            files={"file": ("answer.webm", b"fake-audio", "audio/webm")},
        )
        planned = client.post(f"/api/live-interviews/{session_id}/suggestions")
        suggestion = planned.json()["state"]["live_interview"]["suggestions"][0]
        adopted = client.patch(
            f"/api/live-interviews/{session_id}/suggestions/{suggestion['id']}",
            json={"status": "adopted"},
        )
        answered = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={
                "speaker": "candidate",
                "text": "我用压测和灰度验证方案，P95 延迟下降 30%，异常时可以按灰度批次回滚。",
            },
        )
        answer_segment = answered.json()["state"]["live_interview"]["segments"][-1]
        question_segment = adopted.json()["state"]["live_interview"]["segments"][-1]
        corrected = client.patch(
            f"/api/live-interviews/{session_id}/segments/{answer_segment['id']}",
            json={
                "speaker": "candidate",
                "text": "我用压测和灰度验证方案，P95 延迟下降 40%，异常时可以按灰度批次回滚。",
            },
        )
        confirmed = client.post(
            f"/api/live-interviews/{session_id}/segments/{answer_segment['id']}/evidence",
            json={
                "question_segment_id": question_segment["id"],
                "competency": "System Design",
            },
        )
        duplicate = client.post(
            f"/api/live-interviews/{session_id}/segments/{answer_segment['id']}/evidence",
            json={"competency": "System Design"},
        )
        edit_confirmed = client.patch(
            f"/api/live-interviews/{session_id}/segments/{answer_segment['id']}",
            json={"text": "确认后不应该直接改写"},
        )

    assert denied.status_code == 409
    assert started.status_code == 200
    segment = transcribed.json()["state"]["live_interview"]["segments"][0]
    assert segment["source"] == "asr"
    assert "P95" in segment["text"]
    assert planned.status_code == 200
    assert suggestion["competency"] == "System Design"
    assert suggestion["question_type"] == "follow_up"
    final_live = adopted.json()["state"]["live_interview"]
    assert final_live["suggestions"][0]["status"] == "adopted"
    assert final_live["segments"][-1]["speaker"] == "interviewer"
    assert corrected.status_code == 200
    corrected_segment = corrected.json()["state"]["live_interview"]["segments"][-1]
    assert "40%" in corrected_segment["text"]
    assert confirmed.status_code == 200
    confirmed_state = confirmed.json()["state"]
    assert confirmed_state["live_interview_records"][0]["source"] == "live_interview"
    assert "40%" in confirmed_state["live_interview_records"][0]["answer"]
    assert confirmed_state["live_interview_records"][0]["transcript_segment_ids"] == [
        question_segment["id"],
        answer_segment["id"],
    ]
    assert confirmed_state["evidence"][0]["source"] == "live_interview"
    assert confirmed_state["evidence"][0]["competency"] == "System Design"
    assert duplicate.status_code == 409
    assert edit_confirmed.status_code == 409


def test_live_review_queue_batch_confirms_pending_candidate_answers(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-batch.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/analysis/job",
            json={"session_id": session_id, "text": "System Design Engineer"},
        )
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次架构权衡。"},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我比较了缓存和数据库扩容。"},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我用灰度发布验证并降低了延迟。"},
        )
        confirmed = client.post(
            f"/api/live-interviews/{session_id}/evidence/batch",
            json={"competency": "System Design"},
        )
        empty = client.post(
            f"/api/live-interviews/{session_id}/evidence/batch",
            json={"competency": "System Design"},
        )

    assert confirmed.status_code == 200
    state = confirmed.json()["state"]
    assert len(state["live_interview_records"]) == 2
    assert len(state["evidence"]) == 2
    assert {item["source"] for item in state["evidence"]} == {"live_interview"}
    assert empty.status_code == 409


def test_live_evidence_can_be_revoked_and_reconfirmed(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-revoke.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        question = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次线上故障处理。"},
        ).json()["state"]["live_interview"]["segments"][0]
        answer = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我先止血，再复盘根因并补监控。"},
        ).json()["state"]["live_interview"]["segments"][1]
        confirmed = client.post(
            f"/api/live-interviews/{session_id}/segments/{answer['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "Incident Response"},
        )
        confirmed_state = confirmed.json()["state"]
        record_id = confirmed_state["live_interview_records"][0]["id"]

        revoked = client.delete(f"/api/live-interviews/{session_id}/evidence/{record_id}")
        reconfirmed = client.post(
            f"/api/live-interviews/{session_id}/segments/{answer['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "Incident Response"},
        )

    assert confirmed.status_code == 200
    assert confirmed_state["evidence"][0]["source_record_id"] == record_id
    assert revoked.status_code == 200
    revoked_state = revoked.json()["state"]
    assert revoked_state["live_interview_records"] == []
    assert revoked_state["evidence"] == []
    assert reconfirmed.status_code == 200
    reconfirmed_state = reconfirmed.json()["state"]
    assert len(reconfirmed_state["live_interview_records"]) == 1
    assert len(reconfirmed_state["evidence"]) == 1
    assert reconfirmed_state["live_interview_records"][0]["id"] != record_id


def test_live_candidate_segments_can_be_merged_into_one_evidence_record(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-merge.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        question = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次架构演进。"},
        ).json()["state"]["live_interview"]["segments"][0]
        first_text = "第一阶段我先拆分核心服务，把订单、库存和支付链路分开，并补了基础压测。"
        second_text = "第二阶段我补了监控、灰度发布和回滚预案，最终上线后错误率和延迟指标都比较稳定，以上。"
        first = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": first_text},
        ).json()["state"]["live_interview"]["segments"][1]
        second = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": second_text},
        ).json()["state"]["live_interview"]["segments"][2]
        live = client.get(f"/api/live-interviews/{session_id}").json()["state"]["live_interview"]
        boundary = live["answer_boundary_suggestions"][0]
        assert boundary["question_segment_id"] == question["id"]
        assert boundary["answer_segment_ids"] == [first["id"], second["id"]]
        assert boundary["confidence"] >= 0.75
        assert "已关联最近面试官问题" in boundary["confidence_factors"]
        assert "末段出现回答结束信号" in boundary["confidence_factors"]
        assert live["action_card"]["action_type"] == "merge_boundary"
        assert live["action_card"]["primary_cta"] == "按边界建议合并"

        merged = client.post(
            f"/api/live-interviews/{session_id}/evidence/merge",
            json={
                "segment_ids": [second["id"], first["id"]],
                "question_segment_id": question["id"],
                "competency": "Architecture Evolution",
            },
        )
        duplicate = client.post(
            f"/api/live-interviews/{session_id}/segments/{first['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "Architecture Evolution"},
        )

    assert merged.status_code == 200
    state = merged.json()["state"]
    assert len(state["live_interview_records"]) == 1
    assert len(state["evidence"]) == 1
    record = state["live_interview_records"][0]
    assert record["answer"] == f"{first_text}\n{second_text}"
    assert record["transcript_segment_ids"] == [question["id"], first["id"], second["id"]]
    assert state["live_interview"]["answer_boundary_suggestions"] == []
    assert state["live_interview"]["action_card"]["action_type"] == "plan_gap_question"
    assert state["evidence"][0]["source_record_id"] == record["id"]
    assert duplicate.status_code == 409


def test_live_transcript_dedupes_and_rolls_context_for_question_planning(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-context.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次系统设计经历。"},
        )
        first = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我先梳理瓶颈并设计分层缓存。"},
        ).json()["state"]["live_interview"]
        duplicate = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我先梳理瓶颈并设计分层缓存。"},
        ).json()["state"]["live_interview"]
        for index in range(12):
            client.post(
                f"/api/live-interviews/{session_id}/segments",
                json={
                    "speaker": "candidate",
                    "text": f"补充第 {index} 点：我用指标验证方案并记录风险。",
                },
            )
        planned = client.post(f"/api/live-interviews/{session_id}/suggestions")
        events = client.get("/api/debug/events").json()["events"]

    assert len(first["segments"]) == 2
    assert len(duplicate["segments"]) == 2
    assert duplicate["duplicate_segments_dropped"] == 1
    live = planned.json()["state"]["live_interview"]
    assert live["summarized_until_sequence"] > 0
    assert "请讲一次系统设计经历" in live["rolling_summary"]
    planned_events = [item for item in events if item["action"] == "live_question_planned"]
    assert planned_events
    assert "summary_until=" in planned_events[0]["detail"]
    assert "duplicates_dropped=1" in planned_events[0]["detail"]
    assert "action=merge_boundary" in planned_events[0]["detail"]


def test_live_coverage_guidance_tracks_evidence_gaps(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-guidance.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/analysis/job",
            json={"session_id": session_id, "text": "System Design Engineer"},
        )
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        ).json()["state"]["live_interview"]
        question = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次架构权衡。"},
        ).json()["state"]["live_interview"]["segments"][0]
        first = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我通过压测比较方案并灰度上线。"},
        ).json()["state"]["live_interview"]["segments"][1]
        one_evidence = client.post(
            f"/api/live-interviews/{session_id}/segments/{first['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "System Design"},
        ).json()["state"]["live_interview"]["coverage_guidance"][0]
        second = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我补充了容量评估、回滚预案和故障复盘。"},
        ).json()["state"]["live_interview"]["segments"][-1]
        two_evidence = client.post(
            f"/api/live-interviews/{session_id}/segments/{second['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "System Design"},
        ).json()["state"]["live_interview"]["coverage_guidance"][0]
        client.post(f"/api/live-interviews/{session_id}/suggestions")
        events = client.get("/api/debug/events").json()["events"]

    initial = started["coverage_guidance"][0]
    assert initial["competency"] == "System Design"
    assert initial["priority"] == "high"
    assert initial["evidence_count"] == 0
    assert one_evidence["priority"] == "medium"
    assert one_evidence["evidence_count"] == 1
    assert two_evidence["priority"] == "low"
    assert two_evidence["evidence_count"] == 2
    planned_events = [item for item in events if item["action"] == "live_question_planned"]
    assert planned_events
    assert "top_gap=System Design" in planned_events[0]["detail"]
    assert "top_boundary_confidence=" in planned_events[0]["detail"]


def test_live_question_usage_tracks_blueprint_progress(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-question-usage.db'}")
    app = create_app(
        storage=storage,
        llm_client=BlueprintFallbackQuestionLLM(),
        configure_llm=False,
    )
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        designed = client.post(
            "/api/workflows/enterprise-design",
            json={
                "session_id": session_id,
                "resume_text": "Platform engineer with reliability experience.",
                "job_description": "平台工程师\n岗位职责：系统设计、稳定性治理。\n任职要求：事故复盘。",
                "company_name": "Example",
            },
        ).json()["state"]
        started = client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        ).json()["state"]["live_interview"]
        client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我做过一次缓存架构演进。"},
        )
        planned = client.post(f"/api/live-interviews/{session_id}/suggestions").json()["state"]
        suggestion = planned["live_interview"]["suggestions"][0]
        adopted = client.patch(
            f"/api/live-interviews/{session_id}/suggestions/{suggestion['id']}",
            json={"status": "adopted"},
        ).json()["state"]["live_interview"]
        events = client.get("/api/debug/events").json()["events"]

    assert len(designed["blueprint"]["rounds"][0]["questions"]) == 2
    assert {item["status"] for item in started["question_usage"]} == {"pending"}
    assert [item["status"] for item in planned["live_interview"]["question_usage"]].count(
        "suggested"
    ) == 1
    assert suggestion["source_question_id"]
    assert adopted["used_question_ids"] == [suggestion["source_question_id"]]
    assert adopted["question_usage"][0]["status"] == "pending"
    assert adopted["question_usage"][-1]["status"] == "used"
    assert adopted["question_usage"][-1]["suggested_count"] == 1
    assert planned["live_interview"]["action_card"]["action_type"] == "confirm_evidence"
    assert planned["live_interview"]["action_card"]["primary_cta"] == "确认为证据"
    planned_events = [item for item in events if item["action"] == "live_question_planned"]
    assert planned_events
    assert "pending_blueprint=1" in planned_events[0]["detail"]
    assert "action=confirm_evidence" in planned_events[0]["detail"]


def test_live_evidence_can_be_reevaluated_with_updated_competency(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-reevaluate.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        question = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "interviewer", "text": "请讲一次跨团队项目。"},
        ).json()["state"]["live_interview"]["segments"][0]
        answer = client.post(
            f"/api/live-interviews/{session_id}/segments",
            json={"speaker": "candidate", "text": "我协调平台和业务团队完成迁移。"},
        ).json()["state"]["live_interview"]["segments"][1]
        confirmed = client.post(
            f"/api/live-interviews/{session_id}/segments/{answer['id']}/evidence",
            json={"question_segment_id": question["id"], "competency": "System Design"},
        ).json()["state"]
        record_id = confirmed["live_interview_records"][0]["id"]

        reevaluated = client.patch(
            f"/api/live-interviews/{session_id}/evidence/{record_id}/reevaluate",
            json={"competency": "Cross-team Collaboration"},
        )

    assert reevaluated.status_code == 200
    state = reevaluated.json()["state"]
    assert len(state["live_interview_records"]) == 1
    assert len(state["evidence"]) == 1
    record = state["live_interview_records"][0]
    evidence = state["evidence"][0]
    assert record["id"] == record_id
    assert record["competency"] == "Cross-team Collaboration"
    assert evidence["competency"] == "Cross-team Collaboration"
    assert evidence["source_record_id"] == record_id
    assert record["transcript_segment_ids"] == [question["id"], answer["id"]]


def test_fact_card_decision_api_updates_session_state(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'facts.db'}")
    app = create_app(
        storage=storage,
        llm_client=WorkflowLLM(),
        search_provider=FactSearchProvider(),
        configure_llm=False,
    )
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        analyzed = client.post(
            "/api/analysis/company",
            json={"session_id": session_id, "name": "Example", "context": "Example develops AI systems."},
        ).json()["state"]
        card_id = analyzed["fact_cards"][0]["id"]

        accepted = client.patch(
            f"/api/intelligence/{session_id}/facts/{card_id}",
            json={"action": "accept", "note": "用户确认"},
        )
        rejected = client.patch(
            f"/api/intelligence/{session_id}/facts/{card_id}",
            json={"action": "reject", "note": "来源冲突"},
        )
        reset = client.patch(
            f"/api/intelligence/{session_id}/facts/{card_id}",
            json={"action": "reset"},
        )

    assert accepted.status_code == 200
    assert accepted.json()["state"]["fact_cards"][0]["status"] == "accepted"
    assert accepted.json()["state"]["fact_cards"][0]["note"] == "用户确认"
    assert rejected.status_code == 200
    assert rejected.json()["state"]["fact_cards"][0]["status"] == "rejected"
    assert reset.status_code == 200
    assert reset.json()["state"]["fact_cards"][0]["status"] == "inferred"


def test_live_interviewer_workflow_reaches_sufficient_evidence_evaluation(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'live-sufficient.db'}")
    app = create_app(storage=storage, llm_client=WorkflowLLM(), configure_llm=False)
    with TestClient(app) as client:
        session_id = client.post("/api/interviews/sessions", json={}).json()["id"]
        client.post(
            "/api/workflows/enterprise-design",
            json={
                "session_id": session_id,
                "resume_text": "Platform engineer with stability and architecture experience.",
                "job_description": "平台工程师\n岗位职责：架构设计、稳定性治理。\n任职要求：系统设计、故障复盘。\n团队背景：平台团队。",
                "company_name": "Example",
            },
        )
        client.post(
            f"/api/live-interviews/{session_id}/start",
            json={"consent_confirmed": True},
        )
        entries = [
            ("interviewer", "请讲一次架构权衡。"),
            ("candidate", "我比较了缓存治理、数据库扩容和限流，最终选择先治理缓存。"),
            ("candidate", "上线前通过压测验证 P95 延迟，并保留灰度回滚方案。"),
            ("candidate", "事故后我组织复盘，补充告警、演练和 runbook。"),
        ]
        for speaker, text in entries:
            client.post(
                f"/api/live-interviews/{session_id}/segments",
                json={"speaker": speaker, "text": text},
            )
        state = client.get(f"/api/live-interviews/{session_id}").json()["state"]
        candidates = [
            item for item in state["live_interview"]["segments"] if item["speaker"] == "candidate"
        ]
        client.post(
            f"/api/live-interviews/{session_id}/segments/{candidates[0]['id']}/evidence",
            json={"competency": "System Design"},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments/{candidates[1]['id']}/evidence",
            json={"competency": "System Design"},
        )
        client.post(
            f"/api/live-interviews/{session_id}/segments/{candidates[2]['id']}/evidence",
            json={"competency": "Incident Review"},
        )
        evaluated = client.post(f"/api/evaluations/{session_id}")

    assert evaluated.status_code == 200
    result = evaluated.json()["state"]
    assert len(result["evidence"]) == 3
    assert {item["competency"] for item in result["evidence"]} == {
        "System Design",
        "Incident Review",
    }
    assert result["evaluation"]["recommendation"] != "insufficient_evidence"
