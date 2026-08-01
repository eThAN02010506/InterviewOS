# InterviewOS Architecture

## Overview

InterviewOS is a self-built Python Agent Runtime for interview intelligence.
It does NOT use LangGraph / AutoGen / CrewAI as its core.

## Layers

1. **User Layer** - FastAPI + Web UI
2. **Agent Runtime** - orchestrates agents, manages state, routes messages
3. **Domain Agents** - 9 specialized agents for interview scenarios
4. **LLM Layer** - local LLM via OpenAI-compatible API (Ollama/vLLM)
5. **Tools Layer** - PDF parser, resume parser, web search, whisper, vector search
6. **Data Layer** - SQLite/PostgreSQL + FAISS/Chroma

## Dependency Direction

API routes call `InterviewService`; the service owns session-scoped runtimes and
persists `InterviewState` through `Storage`. Domain agents depend only on core
models and the LLM interface. The composition root in `core.factory` is the only
place that knows every concrete agent type.

Session mutations are serialized with a per-session lock. Different sessions can
still run concurrently, while two requests cannot overwrite the same state.

## Resume Intake and Review

`ResumeProcessor` runs before the Agent workflow. It enforces file and text limits,
extracts PDF/DOCX text, and performs linear-time deterministic checks. The text and
`ResumeReview` are persisted together. Resume claims default to `unverified` and
require a user to confirm, dispute, or request supporting material. Public search
is intentionally separate from upload so candidate data is not disclosed without
an explicit action.

## Workflow Execution

`InterviewService` binds real request inputs to the declarative agent topology.
It persists state before and after every step, so progress and failures survive a
runtime cache miss or process restart. Domain outputs are validated Pydantic models:

- `InterviewStrategy`
- `MockInterviewPlan` and `InterviewQuestion`
- `InterviewBlueprint` and `InterviewRound`

Free-form LLM text is never treated as a completed workflow result. A missing or
invalid required result moves `WorkflowProgress` to `failed` with the current step
and error retained in session state.

## Controlled Autopilot

`AutopilotState` is the control plane above the reusable domain workflows. It records
phase, completed actions, authorization, pause reason, and terminal status. Candidate
autopilot runs intelligence and strategy, starts the interview, evaluates every
answer, and automatically triggers final evaluation. Interviewer autopilot prepares
the evidence-based blueprint and pauses for real interview evidence. It never invents
candidate answers or performs public research without explicit authorization.

Search results carry source quality, official-domain and independent-domain signals.
These are prompt evidence labels rather than truth scores: secondary-source claims
remain attributed, and absence from the public web is not treated as falsehood.

## Mock Interview State Machine

`MockInterviewSession` transitions from `idle` to `active` to `completed`. Only the
current question ID can be answered, preventing duplicate or out-of-order evidence.
`CoachAgent` returns a bounded `AnswerEvaluation`; its four-score average becomes
the confidence of a competency-specific `Evidence` record. Invalid coaching output
does not advance the question index, so the client can safely retry.

The session stores one response and one evidence item per answered question, giving
O(q) time and space over a mock plan of q questions. No transcript or prompt history
is duplicated into the response record.

## Final Evaluation

`EvaluationAgent` separates competency score from evidence confidence and aggregates
only persisted evidence. `FeedbackAgent` derives two views from the same structured
result: actionable candidate coaching and evidence-aware interviewer notes. The
service completes the interview stage only after both outputs validate; an empty
evidence set is rejected before any LLM call.

## Runtime Settings

The settings API changes a shared mutable search-provider router, so existing and
future Agent runtimes observe provider changes immediately. API keys are write-only
and retained in process memory; environment variables are the persistent startup
source. This avoids storing plaintext credentials in the application database.

## Local Web Application

The product UI is a dependency-free static application served by FastAPI from the
same origin as the API. This is deliberate: InterviewOS targets local models and
LAN services, so a separately hosted frontend would complicate connectivity and
secret handling without adding product value. The UI consumes documented API
routes and stores only the selected session ID in browser-local storage.

## Debug Observability

`DebugEventStore` is an in-memory ring buffer with O(1) append and O(capacity)
filtered reads. `AgentRuntime` records lifecycle timing without duplicating prompts.
Debug endpoints are read-only except for a fixed `/models` connectivity probe and
reject non-loopback clients. The console is observability, not a remote execution
surface.

## Agent Communication

Agents communicate via Messages through the Runtime.
The Runtime maintains a global InterviewState that all agents read and mutate.

## Key Design Decisions

- **Evidence-based evaluation**: every assessment is backed by Evidence records
- **Three-way fusion**: Candidate + Job + Interviewer → Strategy
- **State machine**: InterviewStage tracks where in the process we are
