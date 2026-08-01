# InterviewOS

> A Local LLM-powered Interview Intelligence Operating System

Based on a Python self-built domain-specific Agent Runtime for interview intelligence.

## Quick Start

```bash
pip install -e ".[dev,local-llm]"
cp .env.example .env
uvicorn interview_os.api.app:app --reload
```

Open `http://127.0.0.1:8000` for the local product UI. It includes candidate
preparation, enterprise interview design, interactive mock interviews, runtime
settings, and the Debug Console.

The API now persists session state in SQLite. A minimal analysis flow is:

PDF and Word (`.docx`) resumes can be uploaded through
`POST /api/resumes/{session_id}/upload`. Files are parsed locally before the Agent
workflow runs. The response includes extraction warnings and human-review items
for education, employment, certifications, and measurable claims. Uploading a
resume never starts public web research automatically.

1. `POST /api/interviews/sessions`
2. `POST /api/analysis/resume` with the returned `session_id`
3. `POST /api/analysis/job` and `POST /api/analysis/interviewer`
4. `GET /api/interviews/sessions/{session_id}` to retrieve the accumulated state

For an end-to-end flow, call one of:

- `POST /api/autopilot/{session_id}/run` — automatically executes every authorized
  analysis and planning step. Candidate mode starts the AI-led interview, evaluates
  each answer, and generates the final dual-side report after the last answer.
  Autopilot pauses instead of fabricating candidate answers or real-world evidence.

- `POST /api/workflows/candidate-prep` — candidate, job, company and optional
  interviewer analysis, followed by a structured strategy and mock interview plan.
- `POST /api/workflows/enterprise-design` — candidate, job and company analysis,
  followed by a structured multi-round interview blueprint.

Workflow progress is stored under `state.workflow`. Clients can poll the existing
session endpoint while a workflow request is running. Invalid structured LLM output
marks the workflow as `failed` and preserves the failing step and error message.

After candidate preparation, run an interactive mock interview:

1. `POST /api/mock-interviews/{session_id}/start`
2. Present the returned `current_question`
3. `POST /api/mock-interviews/{session_id}/answers` with its `question_id` and answer
4. Repeat until `mock_session.status` is `completed`

Each answer receives validated 0–1 scores for content, technical depth, structure,
and impact. The average becomes evidence confidence for the question competency;
observed and missing signals remain attached to the persisted answer and evidence.

When evidence is available, `POST /api/evaluations/{session_id}` runs the final
evaluation and feedback workflow. Candidate UI presents strengths, improvements,
and an action plan; interviewer UI presents competency scores, evidence confidence,
signal gaps, and a hiring recommendation. Missing evidence returns `409` instead of
fabricating a conclusion.

## Web research

Company and interviewer analysis can enrich prompts with public search evidence.
Configure one provider:

```bash
TAVILY_API_KEY=tvly-your-key
# or
SEARXNG_BASE_URL=http://localhost:8080
# or
BRAVE_SEARCH_API_KEY=your-key
```

When more than one is configured, precedence is Tavily, SearXNG, then Brave.
SearXNG remains available for a fully self-hosted deployment and requires its JSON
response format to be enabled. Search results retain their title, URL, snippet,
and provider so the analysis can be traced back to public sources.

Runtime search and Local LLM settings can also be changed at
`http://127.0.0.1:8000/api/settings/ui`. Secrets are write-only and are never
returned to the browser. UI changes take effect immediately for existing sessions;
environment variables remain the persistent startup configuration.

Runtime secrets are intentionally not written to SQLite or returned by the API.
For persistence across restarts, provide them through the process environment or a
secret manager. OS-keychain persistence can be added later without changing the
settings API.

## Debug Console

The UI includes a read-only operational console backed by `/api/debug`. It shows
bounded Agent lifecycle events, durations, workflow failures, session summaries,
and a fixed LLM connectivity probe. The console:

- accepts requests only from localhost;
- stores at most `DEBUG_EVENT_CAPACITY` events (500 by default);
- truncates message output and never exposes API keys;
- does not provide arbitrary Python, shell, SQL, or prompt execution.

Do not reverse-proxy `/api/debug` to untrusted networks without authentication.

## Development Phases

- Phase 1: Core Runtime (Agent / State / Memory / Tool)
- Phase 2: Recruitment Analysis (Resume / JD / Interview Designer)
- Phase 3: Candidate Intelligence (Interviewer Analysis / Company Analysis / Mock Interview)
- Phase 4: Real-time Assistance (Whisper / Live Copilot)
