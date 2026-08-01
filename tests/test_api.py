from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from interview_os.api.app import create_app
from interview_os.database.storage import Storage
from interview_os.models.local_llm import LocalLLMClient


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
        raise AssertionError(prompt)

    async def embed(self, text):
        return []


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
    app = create_app(storage=storage, llm_client=llm, configure_llm=False)
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
    assert response.json()["state"]["evaluation"]["recommendation"] == "hire"
    assert response.json()["state"]["feedback"]["overall"] == "Meets the bar"
