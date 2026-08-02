# InterviewOS

> Local-first interview intelligence for candidates, interviewers, and live
> interview execution.

InterviewOS is a Python + Local LLM Agent system for recruitment and interview
scenarios. Its long-term product direction is not just "generate interview
questions"; it is an AI interview operating system that can understand the
candidate, job, company, interviewer, public evidence, and live conversation, then
help the human interviewer run a structured, evidence-grounded interview.

The product deliberately keeps a human confirmation boundary. AI can analyze,
prepare, listen, summarize, suggest, and score against evidence, but it must pause
for real candidate input, explicit consent, entity corrections, resume claim review,
and final interviewer decisions.

## Quick Start

```bash
pip install -e ".[dev,local-llm]"
cp .env.example .env
uvicorn interview_os.api.app:app --reload
```

Open `http://127.0.0.1:8000` for the local product UI. It includes candidate
preparation, enterprise interview design, interactive mock interviews, runtime
settings, and the Debug Console.

The interviewer workspace includes a **Live Interview Copilot**. With explicit
consent, it processes typed or recorded interview turns and prepares the next
follow-up or main question. It is a decision-support surface: the interviewer
chooses, edits, skips, or postpones every suggested question. See the
[product requirements](docs/product_requirements.md) for scope, privacy rules,
delivery phases, and acceptance criteria.

## Development Working Agreement

This project is being developed as a usable local product rather than a throwaway
demo. For every meaningful implementation batch, update this README and
`docs/product_requirements.md` together with the code change, run the relevant
validation checks, and create a descriptive git commit. Commit messages should
capture the user-facing capability, the major technical changes, and any important
validation result or known limitation.

When a requested change touches the UI or API, restart the local service after the
commit so the browser reflects the committed state. Secrets, local API keys,
resume content, and live transcript details must stay out of commit messages,
debug logs, screenshots, and exported data.

## Current Product Shape

InterviewOS currently has two primary UI modes:

- Candidate mode: resume upload, JD/company/interviewer analysis, preparation
  strategy, mock interview, scored answers, and improvement report.
- Interviewer mode: interview design, candidate evidence review, live interview
  copilot, debug console, and hiring evaluation once enough confirmed evidence
  exists.

The system is built around a few product rules:

- Public research enriches company and interviewer understanding, but searched
  identity corrections require user confirmation before changing the canonical
  entity. The confirmation dialog also allows a user-edited canonical name instead
  of forcing the searched alias.
- Resume parsing can identify suspicious or incomplete claims; downstream Agents
  should rely on confirmed or explicitly accepted facts.
- JD quality matters. A title-only JD is treated as insufficient context, and the
  UI should ask for responsibilities, requirements, and team background while
  separating explicit requirements from AI assumptions. Inferred requirements can
  be confirmed, edited into explicit requirements, or removed before later Agents
  use them.
- Search conclusions are organized into fact cards with source URLs and status:
  verified, inferred, conflicting, or needs review. The UI groups cards by conflict,
  company, interviewer, technology, and public-opinion categories. Each card shows
  source count, best source quality, provider fetch time, cache-hit status,
  filtering reason, and generation time. Users can confirm a card for later
  context, reject it, or reset it to inferred review state.
- Secrets and sensitive resume/transcript content must not appear in logs, SQLite
  exports, settings responses, or the Debug Console.

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
settings are stored in a local permission-restricted file and environment variables
can still provide startup defaults.

Runtime secrets are not written to SQLite, returned by the API, or included in the
Debug Console. A platform keychain or external secret manager can replace the local
settings store later without changing the settings API.

## Debug Console

The UI includes a read-only operational console backed by `/api/debug`. It shows
bounded Agent lifecycle events, durations, workflow failures, session summaries,
and a fixed LLM connectivity probe. The console:

- accepts requests only from localhost;
- stores at most `DEBUG_EVENT_CAPACITY` events (500 by default);
- truncates message output and never exposes API keys;
- does not provide arbitrary Python, shell, SQL, or prompt execution.

Do not reverse-proxy `/api/debug` to untrusted networks without authentication.

## Live Interview Copilot

The turn-based live copilot is implemented. It records one speaker turn in the
browser, sends it to the configured LAN ASR, and asks the configured text model for
a grounded next question. Typed/pasted dialogue remains available when audio fails.
The live evidence review loop is also implemented: transcript segments can be
edited before confirmation, candidate answers can be converted into traceable
`live_interview` evidence, and the review queue shows whether enough evidence and
competency coverage exists to generate a hiring recommendation. Confirmed live
evidence can be revoked without deleting the transcript segment, so the interviewer
can correct speaker/text and confirm the answer again. Multiple adjacent candidate
segments can also be merged into one answer before scoring, which is important for
chunked ASR output and longer responses. The review queue now proposes conservative
answer-boundary suggestions for consecutive unarchived candidate segments after the
same interviewer question; the interviewer still confirms before any evidence is
created. Confirmed live records can also be re-evaluated with an updated question
or competency while preserving the same transcript trace.

A conservative **chunked continuous listening MVP** is also implemented. It reuses
the existing HTTP ASR upload endpoint instead of introducing a second protocol too
early. The browser records short chunks, uploads them sequentially, shows queue and
upload status, and marks each chunk as `unknown` speaker by default. This keeps the
interviewer in control: the user must correct speaker/text and explicitly confirm
candidate answers before any chunk can become hiring evidence.
For longer interviews, the live state now maintains a bounded rolling transcript
summary and drops exact repeated transcript chunks before they can grow the
conversation context. The next-question planner receives the rolling summary,
recent confirmed turns, the interview blueprint, and confirmed evidence instead of
the full raw transcript.
The competency map also provides deterministic coverage guidance: for each target
competency it shows evidence count, strongest confidence, priority, why the signal
is still weak or sufficient, and a sample question the interviewer can ask next.

The product direction for live interviews is an interviewer-side copilot that can
listen during the conversation, keep an evidence map, and quietly prepare the next
question. It should help the interviewer stay structured without taking over the
conversation. The AI may suggest, summarize, detect gaps, and prepare follow-ups;
the interviewer remains responsible for choosing what is asked and for confirming
what becomes evidence.

The remaining roadmap is deliberately separated:

1. **Completed turn-based MVP** — live state, typed or recorded turns, structured
   planning, visible consent controls, deterministic fallbacks, interviewer
   decisions, transcript review, evidence confirmation, and final evaluation gates.
2. **Completed chunked continuous listening MVP** — keep the existing HTTP ASR upload path,
   let the browser record short sequential audio chunks, mark uncertain chunks as
   `unknown` speaker by default, and require transcript review before evidence is
   created. This gives a practical bridge before true streaming.
3. **Completed bounded context MVP** — exact duplicate transcript chunks are dropped,
   older stable turns are summarized into bounded context, and next-question planning
   uses summary + recent turns + evidence rather than the full transcript.
4. **Continuous streaming** — partial transcript events, stronger answer-boundary
   detection, WebSocket reconnect/protocol-level deduplication, and optional speaker diarization.
5. **Evidence map hardening** — confirmed turns already become `live_interview`
   evidence, can be revoked, merged, re-evaluated, and turned into coverage
   guidance; next work is better boundary confidence.
6. **Hardening** — long-interview tests, deterministic fallbacks, latency budgets,
   redacted observability, and cost measurement.

The live path must not invoke an LLM for every partial word. Stable transcript turns
feed a bounded context made from recent dialogue, a rolling summary, the interview
blueprint, and confirmed evidence. The target is transcript feedback within two
seconds and a next-question suggestion within five seconds after an answer ends.

The default LAN ASR integration is `http://192.168.1.97:8003` (MiMo-V2.5-ASR),
behind a configurable provider adapter rather than hard-coded into the interview
workflow. Its `/health`, OpenAI-compatible `/v1/audio/transcriptions`, WAV input,
and JSON text response have been verified. Streaming support and concurrency limits
remain part of long-interview hardening. When ASR is unavailable, the live workspace
retains typed/pasted transcript input as the safe fallback.

`http://192.168.1.8:9001` was tested with the same multi-sentence WAV fixture. Its
OpenAI-compatible endpoint was faster, but returned only the first sentence, so it
is not the default evidence source until that truncation behavior is resolved.

## Next Implementation Plan

The next work should move in this order:

1. **True streaming design** — add WebSocket transcript events, stronger answer-boundary
   detection, reconnect handling, and protocol-level deduplication.
2. **Interviewer-side long-run pass** — repeat the verified interviewer workflow
   with longer 60-90 minute transcripts, mixed competencies, ASR failures, and model
   retries.
3. **Search and fact-card hardening** — improve source ranking, richer conflict
   explanations, Tavily cache-hit reasoning, and longer-lived source review UX.
4. **Operational hardening** — expand Debug Console timings, retry paths, redacted
   cost/token metrics, local secret persistence tests, and 60-90 minute live
   interview load tests.
