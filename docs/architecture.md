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

Candidate input transitions are centralized before resume analysis and both workflow
variants. With no substantive interview activity, a new resume atomically invalidates
all candidate-derived strategy, blueprint, mock, live, evidence, and report state.
Once answers, transcript segments, evidence, or a recording exists, regeneration is
rejected with a conflict and the caller must create a new session. Slow document
parsing and optional LLM structuring run outside the session mutation lock.

## Controlled Autopilot

`AutopilotState` is the control plane above the reusable domain workflows. It records
phase, completed actions, authorization, pause reason, and terminal status. Candidate
autopilot runs intelligence and strategy, starts the interview, evaluates every
answer, and automatically triggers final evaluation. Interviewer autopilot prepares
the evidence-based blueprint and pauses for real interview evidence. It never invents
candidate answers or performs public research without explicit authorization.
The authorization bit is explicitly forwarded into each nested workflow instead of
relying on defaults. Resume upload preserves the session display identity, while a
document-explicit name may still replace it during grounded candidate analysis.

Agent runtime logs are metadata-only (`agent`, input/output character counts,
duration, message type, and exception class). Prompts, resumes, transcripts, model
response bodies, provider error text, and credentials are excluded from logs and
debug events.

Structured agent calls use temperature `0.2`. `LocalLLMClient` detects `gpt-oss`
models and supplies `chat_template_kwargs.reasoning_effort=low` unless explicitly
overridden. This is the field consumed by llama.cpp's GPT-OSS Jinja template and keeps
the model's hidden reasoning from consuming the complete token budget before the
OpenAI-compatible response emits final JSON content.

`CoachAgent` does not trust model-written replacement answers. The model returns only
scores plus bounded feedback/signals. A deterministic post-validation layer displays
the verbatim submitted answer inside a STAR completion scaffold; legacy model rewrites
are discarded, with an explicit warning when they contain unsupported numeric facts.
When model scoring JSON remains invalid, a bounded deterministic rubric derives scores
from observable answer features (detail length, action/structure/result markers, and
verified metrics). It records a degradation event and never presents the result as an
equivalent substitute for human review.

`LocalLLMClient.chat` separates transport recovery from model-output recovery. One
bounded retry handles connection/timeouts and HTTP 5xx responses; only successfully
returned model content enters the agent's structured JSON retry loop. Exhausted
transport failures become a constant redacted sentinel and metadata-only log event.

Search results carry source quality, official-domain and independent-domain signals.
These are prompt evidence labels rather than truth scores: secondary-source claims
remain attributed, and absence from the public web is not treated as falsehood.
Official-domain matching considers every quoted entity in a joint person/company
query. Fact-card extraction discards image captions, navigation boilerplate, and
title-only fallbacks; these remain auditable in the underlying source list.

## Mock Interview State Machine

`MockInterviewSession` transitions from `idle` to `active`, through `evaluating`
when answers require a final report, and finally to `completed`. Evaluation failure
returns it to `active` without dropping responses, so finish is safely retryable. Its
question pool is append-only and refilled in bounded batches while the current
question index acts as a navigation cursor. Only the current question ID can be
answered; an already answered question requires explicit retry bound to the exact
response ID, which replaces its response and linked evidence without confusing a
follow-up with its parent. Retrying a main response invalidates child follow-ups and
their evidence because they were elicited from the superseded answer. `CoachAgent` returns a bounded `AnswerEvaluation`;
its four-score average becomes evidence confidence.
After submission, the client renders the latest response's persisted `question` and
`is_follow_up` fields beside its evaluation. It does not derive that label from the
already-cleared pending-follow-up state or the parent question cursor.

Model refills run outside the session lock and append under the lock. A unique local
question bridges an exhausted pool immediately, and persisted in-flight flags are
cleared when a runtime is restored because process-local tasks cannot survive a
restart. The same restoration changes an orphaned `evaluating` state back to `active`
so finalization can be retried. State grows linearly with the number of questions actually visited and is
bounded operationally by the user-controlled interview duration rather than by an
initial fixed plan.

## Final Evaluation

`EvaluationAgent` separates competency score from evidence confidence and aggregates
only persisted evidence. `FeedbackAgent` derives two views from the same structured
report, then enforces a role boundary: candidate `overall` cannot contain hiring
language and candidate action items cannot contain interviewer-to-candidate commands.
Recruitment recommendations and verification notes remain in interviewer-only fields.
The result is actionable candidate coaching and evidence-aware interviewer notes. The
service completes the interview stage only after both outputs validate; an empty
evidence set is rejected before any LLM call.
Before persistence, `EvaluationAgent` recomputes the overall score from competency
scores and maps recommendation labels through fixed score/confidence thresholds.
`strong_hire` requires score >= 0.85 and mean confidence >= 0.75; conflicting model
labels are replaced and the calibration is retained as a report risk.
Provisional deterministic answer scores form a hard decision boundary: if any remain
unreviewed, the recommendation is `insufficient_evidence` even when the numeric score
would otherwise cross a hire threshold. Feedback post-validation also relabels missing
descriptions from “negative evidence” to “pending verification”.
`FeedbackAgent` then replaces free-form recommendation prose with a deterministic
explanation of the final calibrated enum and score. This makes the decision header
and reasoning one atomic, internally consistent view.

## Runtime Settings

The authenticated settings API changes shared mutable provider clients, so existing
and future Agent runtimes observe changes immediately. API keys are write-only and
stored outside SQLite in a permission-restricted, atomically replaced local JSON
file; environment variables remain startup defaults. Settings responses and Debug
events expose only configuration status, never secret values.

Updates acquire one application-level async lock and snapshot every mutable provider
before applying changes. Any validation or persistence error triggers a best-effort
rollback of all providers, preventing concurrent requests or a late ASR error from
leaving an earlier search/LLM mutation active. The resume client aliases the main
LLM only until its first dedicated configuration; that update constructs a separate
client, atomically swaps the service reference, and participates in rollback and
shutdown like every other transport.

## Local Web Application

The product UI is a dependency-free static application served by FastAPI from the
same origin as the API. This is deliberate: InterviewOS targets local models and
LAN services, so a separately hosted frontend would complicate connectivity and
secret handling without adding product value. The UI consumes documented API
routes and stores the bearer token, selected role, and selected session ID in
browser-local storage. Business and settings APIs require authentication; the Debug
API additionally rejects non-loopback clients.

## Debug Observability

`DebugEventStore` is an in-memory ring buffer with O(1) append and O(capacity)
filtered reads. `AgentRuntime` records lifecycle timing without duplicating prompts.
Debug endpoints are read-only except for a fixed `/models` connectivity probe and
reject non-loopback clients. Every event captures the request owner when it is written,
including search and background events without a session ID; event reads, session
lists, and session details are filtered by that owner. Owner/session/level predicates
are evaluated before the result limit, preventing another account's newer events from
starving the current account's diagnostics. Localhost is not an authorization boundary.
The console is observability, not a remote execution surface.

## Streaming Suggestions

Text and omni model adapters use HTTPX streaming contexts so upstream SSE bytes are
consumed before the completion finishes. The API relays JSON-encoded `append` and
`replace` SSE records, preserving embedded newlines, and persists the assembled
suggestion after the stream completes. A transport failure after partial output emits
a redacted deterministic replacement, so neither browser text nor persisted state
contains the internal error or incomplete suggestion. Raw browser `fetch` calls include the same bearer token
as ordinary API requests; disconnecting closes the response generator and upstream
HTTP stream. Suggestion write-back rechecks that the live session is still active,
discarding completions that arrive after pause or finish.
Because many compatible streaming servers omit final usage metadata, the text
adapter estimates prompt tokens once the upstream accepts the request and estimates
completion tokens from the assembled stream. Failed connections before response
acceptance do not inflate prompt-token or cost totals.

Tracked background work is an application-lifecycle dependency: shutdown first
cancels scoring and mock-refill tasks, then closes each distinct model/ASR transport
once, and closes storage last. Shutdown errors are reduced to exception types and do
not prevent later resources from being released; rejected post-close coroutines are
explicitly closed rather than left for Python to warn about during collection.

## Agent Communication

Agents communicate via Messages through the Runtime.
The Runtime maintains a global InterviewState that all agents read and mutate.

## Key Design Decisions

- **Evidence-based evaluation**: every assessment is backed by Evidence records
- **Three-way fusion**: Candidate + Job + Interviewer → Strategy
- **State machine**: InterviewStage tracks where in the process we are
