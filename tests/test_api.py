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
        raise AssertionError(prompt)

    async def embed(self, text):
        return []


def test_session_resume_analysis_flow(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    app = create_app(storage=storage, llm_client=MockLLM(), configure_llm=False)

    with TestClient(app) as client:
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


def test_missing_session_returns_404(tmp_path):
    storage = Storage(f"sqlite+aiosqlite:///{tmp_path / 'missing.db'}")
    app = create_app(storage=storage, llm_client=MockLLM(), configure_llm=False)
    with TestClient(app) as client:
        response = client.get("/api/interviews/sessions/missing")
    assert response.status_code == 404


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
