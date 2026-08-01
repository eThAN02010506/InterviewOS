# InterviewOS

> A Local LLM-powered Interview Intelligence Operating System

Based on a Python self-built domain-specific Agent Runtime for interview intelligence.

## Quick Start

```bash
pip install -e ".[dev,local-llm]"
cp .env.example .env
uvicorn interview_os.api.app:app --reload
```

The API now persists session state in SQLite. A minimal analysis flow is:

1. `POST /api/interviews/sessions`
2. `POST /api/analysis/resume` with the returned `session_id`
3. `POST /api/analysis/job` and `POST /api/analysis/interviewer`
4. `GET /api/interviews/sessions/{session_id}` to retrieve the accumulated state

For an end-to-end flow, call one of:

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

## Development Phases

- Phase 1: Core Runtime (Agent / State / Memory / Tool)
- Phase 2: Recruitment Analysis (Resume / JD / Interview Designer)
- Phase 3: Candidate Intelligence (Interviewer Analysis / Company Analysis / Mock Interview)
- Phase 4: Real-time Assistance (Whisper / Live Copilot)
