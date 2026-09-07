# InterviewOS Architecture

## Overview

InterviewOS is a self-built Python Agent Runtime for interview intelligence,
delivered through one shared Web UI in either a browser or a packaged Tauri
desktop application. It does NOT use LangGraph / AutoGen / CrewAI as its core.

## Layers

1. **Client Runtime** - the static Web UI plus either the browser host or the
   Tauri WebView and Rust launcher
2. **Application/API** - FastAPI routes, authentication, validation, and the
   `InterviewService` facade
3. **Agent Runtime** - orchestrates agents, manages state, and routes messages
4. **Domain Agents** - 11 specialized agents for interview scenarios
5. **Model Layer** - OpenAI-compatible text, ASR, TTS, and optional omni clients
6. **Tools Layer** - document parsing, public web search, and retrieval helpers
7. **Data Layer** - SQLAlchemy storage with SQLite by default, plus local
   settings, metrics, debug, search-cache, and recording files

## Dependency Direction

API routes call `InterviewService`; the service owns session-scoped runtimes and
persists `InterviewState` through `Storage`. Domain agents depend only on core
models and the LLM interface. The composition root in `core.factory` is the only
place that knows every concrete agent type.

The Web UI imports a single transport boundary from `web/modules/api.js`.
`web/modules/runtime.js` resolves a same-origin browser endpoint or a
launcher-injected desktop endpoint, so feature controllers do not need to know
where FastAPI is running. JSON, text, blob, and streaming consumers all use the
same authentication, cancellation, and stale-account/session checks.

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
The post-validator also rebuilds four behavior-anchored feedback cards from the
submitted answer: role evidence, decision depth, structure, and result/reflection.
Each card contains an observable feature and one concrete next step; model prose
cannot create the evidence explanation. The dimensions are equally weighted until a
validated job-specific weighting policy exists.

`LocalLLMClient.chat` separates transport recovery from model-output recovery. One
bounded retry handles connection/timeouts and HTTP 5xx responses; only successfully
returned model content enters the agent's structured JSON retry loop. Structured
calls use temperature zero plus a Pydantic-derived `json_schema` response format.
Endpoints that reject the optional hint with 400/404/422 are capability-cached and
retried without it. A validation failure is reduced to field locations and error
types; the second attempt receives that compact repair instruction, while Debug only
stores the reason, attempt, character count, and structured mode. Raw model text and
candidate content are never persisted. Exhausted transport failures become a
constant redacted sentinel and metadata-only log event.

Search results carry source quality, official-domain and independent-domain signals.
These are prompt evidence labels rather than truth scores: secondary-source claims
remain attributed, and absence from the public web is not treated as falsehood.
Official-domain matching considers every quoted entity in a joint person/company
query. Fact-card extraction discards image captions, navigation boilerplate, and
title-only fallbacks; these remain auditable in the underlying source list.
For a title-only JD, authorized public research runs before the parallel Job Agent
step. Only title-relevant results become prompt context. The resulting requirements
are persisted as inferred with their source URLs while `job.raw_description` retains
the user's original title. No consent or no relevant result leaves the generic title
path intact.

`SearchProviderManager` generations are identified by a SHA-256 fingerprint of the
effective provider and credential without exposing that credential. The fingerprint,
query, limit, and depth form both the cache and process-local single-flight key, so
identical concurrent lookups share one provider request. A settings switch advances
the generation; an old request may complete for its original caller but the
generation/fingerprint check prevents it from writing into the new cache. Cache and
metrics files use bounded permission-restricted JSON stores, and persistence failures
are logged without changing the already successful search result.

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
its four-score average becomes evidence confidence. Before persistence,
`spoken_answer` keeps raw and cleaned transcripts separate, classifies the answer
type, extracts ordered steps with source excerpts, derives question coverage, and
applies downward-only score caps when model scores exceed observable evidence.
Calibration reasons remain in `AnswerEvaluation.spoken_analysis` for UI audit.
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

Mock audio has three separate data paths. ASR previews are cumulative, rate-limited
snapshots and never mutate session state. The stopped recording is transcribed once,
atomically staged under a session-scoped opaque ID, and bound to one response only
when the answer is submitted. The browser uses a local object URL immediately; after
a refresh it fetches the file through an owner-checked endpoint. The upload boundary
accepts only signature-validated WAV/WebM/M4A data. Retry performs model work before
mutating state, then atomically moves superseded responses into `attempt_history` and
replaces their active Evidence; prior recordings remain available for comparison or
explicit deletion. Startup and per-save cleanup remove only expired unbound files and
protect both active and archived recordings. Every callback captures its source
session and generation; session changes stop tracks, revoke URLs, and invalidate late
ASR responses. TTS accepts only the current owned question ID, optionally resolves an
exact persisted response question, and returns audio without persisting it. An abort
controller plus session/question identity check prevents stale TTS races. Optional omni delivery analysis
is scheduled after stable transcription so the HTTP transcription response is not held
for its 25-second timeout. A process-local status cache supports polling, and completed
feedback is attached to an active or archived response when one already references the
recording. It may discuss only changeable speaking behavior;
its result also passes a server-side prohibited-inference filter. Rejected or failed
output degrades to deterministic coaching and is never an Evidence input.

Whole-session live audio is a separate immutable archive. Each recording period is
uploaded as a `LiveAudioPart` with a client-generated idempotency ID, byte count,
SHA-256 digest, and timestamp. `audio_parts` is the authoritative ordered manifest;
`audio_file` remains only the latest-part pointer for backward-compatible clients.
An active recording rotates after five minutes or 64 MiB. Rotation starts a successor
`MediaRecorder` against the same `MediaStream` before stopping its predecessor; the
small overlap avoids a speech gap, while sequence numbers impose FIFO persistence even
if stop callbacks arrive out of order. Each part retains the recording-run archive
revision. A three-part/192 MiB high-water mark stops capture and retains queued bytes
instead of growing memory without bound. Explicit pause, session/auth switch, finish,
and desktop shutdown seal and drain the recorder/queue. `pagehide` may make only a
small best-effort status mutation and never uploads a large audio body.

Archive append requires `recording_id` plus `expected_audio_revision`. Whole-archive
delete requires the expected revision and a UUID `operation_id`; its persisted receipt
makes a lost-response retry idempotent. The `audio_archive_revision` tombstone fences a
delayed upload after explicit deletion.

Audio-file writes use session-scoped temporary files and atomic replacement.
Legacy-file replacement and archive deletion use `.replace-pending` and `.delete-pending`
journals so the database manifest determines whether recovery restores or removes a
file. Startup reconciles journals, orphan files, and the persisted manifest; an
uncertain database commit marks the session for the same reconciliation on its next
locked access. Downloads first validate ownership and the persisted digest, then
stream an already-open descriptor so a concurrent rename cannot switch the served
bytes.

Multipart ingress is bounded twice. `MultipartBodyLimitMiddleware` counts both
Content-Length and streamed ASGI body bytes before an endpoint can let Starlette's
multipart spool grow beyond the file limit plus 1 MiB of form overhead. After
authentication, upload routes resolve the owner/session before copying from that spool
and use a shared 1 MiB bounded reader. Whole-session capture rotates at 64 MiB;
the archive route accepts at most 65 MiB of file content and its wire boundary is
66 MiB including multipart overhead. Live/mock final audio is limited to 25 MiB,
and ASR preview and resume input to 10 MiB.
These limits bound wire/temp-disk and application-heap materialization separately.

Live state transitions use a second optimistic boundary independent of the audio
archive. Start/resume compare `status_revision`, increment `capture_epoch` only for a
new active period, and persist the winning client UUID as
`active_start_operation_id`. A repeated call with that UUID confirms an uncertain
commit without another increment; a competing operation conflicts. Pause/completion
clear the active owner so later capture registration cannot claim a retired epoch.

Answer scoring records its input modality (`typed`, `asr`, or `live_asr`) and uses the
versioned `evidence-v3` bilingual, cross-domain deterministic analyzer around the model
draft. One question contract drives pre-answer requirements, example validation,
post-answer coverage, score calibration, and feedback. The analyzer stores raw and
cleaned text, semantic steps, direct-topic relevance, question coverage, and
pre-calibration scores. It separates forecasts/baselines/targets from observed outcomes
and preserves adjacent metric clauses. Calibration is downward-only; unrelated answers
receive relevance caps, while complete short follow-ups are not expanded into unasked
STAR requirements. Ambiguous typed transition words never trigger a speech penalty,
while strong ASR fillers/repeated repairs may cap only the structure dimension. This
metadata travels with the answer and makes every final score adjustment auditable.

Public title-only JD context is an internal, untrusted prompt input. It is filtered
by title relevance, labeled inferred, and never replaces the user's stored raw JD.
Both successful and failed workflows run the same provenance attachment step, so a
later Agent failure cannot leave source-enriched prompt text in persisted state.

## Final Evaluation

`EvaluationAgent` separates competency score from assessment confidence and rebuilds
every competency field, supporting signal, gap, summary, and risk from persisted
`Evidence` rows. Final evaluation and feedback are deterministic and make no model
call. `FeedbackAgent` derives two views from the same structured report, then enforces
a role boundary: candidate `overall` cannot contain hiring language and candidate
action items cannot contain interviewer-to-candidate commands.
Recruitment recommendations and verification notes remain in interviewer-only fields.
The result is actionable candidate coaching and evidence-aware interviewer notes. The
service completes the interview stage only after both outputs validate; an empty
evidence set is rejected before any LLM call.
`EvaluationAgent` computes the overall score from competency scores and maps
recommendation labels through fixed score/confidence thresholds.
`strong_hire` requires score >= 0.85 and mean confidence >= 0.75; conflicting model
labels are replaced and the calibration is retained as a report risk.
The model-facing `AnswerEvaluationDraft` deliberately excludes trust metadata. The
server constructs the persisted `AnswerEvaluation` and owns its structured
`scoring_source` and `review_status` provenance.
Provisional deterministic answer scores and unfinished live scoring form a hard
decision boundary: if any remain unreviewed, the recommendation is
`insufficient_evidence` even when the numeric score would otherwise cross a hire
threshold. A compatibility validator migrates known pre-provenance rule-score records
once during state loading. A human review API and UI can replace the rubric scores,
classify linked evidence polarity, and invalidate stale final reports. Background live
scoring is a pure calculation until it reacquires the session lock; a late AI result is
dropped when the persisted record is already `human/reviewed`.

Negative interviewer notes are emitted only for Evidence explicitly classified
`negative` by a human reviewer, repeat the exact signal, and include its Evidence ID.
Missing evidence remains a verification gap. `FeedbackAgent` deterministically explains
the final calibrated enum and score, keeping the decision header and reasoning atomic.

## Runtime Settings

The authenticated settings API changes shared mutable provider clients, so existing
and future Agent runtimes observe changes immediately. API keys are write-only and
stored outside SQLite in a permission-restricted, atomically replaced local JSON
file; environment variables remain startup defaults. Settings responses and Debug
events expose only configuration status, never secret values.

The settings file is treated as untrusted input. Reads require a regular file no
larger than 1 MB, reject non-finite numbers, unpaired Unicode surrogates, and JSON
nesting beyond 64 levels, and validate every known section with its API schema.
Persisted provider endpoints are accepted at startup only when they satisfy the
shared strict URL parser: exact whitespace-free HTTP(S), hostname and valid port,
with no embedded userinfo, query, or fragment. LAN hosts and IP literals are valid.
An invalid or unreadable file is left untouched for manual recovery instead of being
silently overwritten with defaults.

`LocalSettingsStore` serializes readers and writers across threads and processes,
creates randomly named private temporary files, flushes them before `os.replace`,
and removes only temporary files matching its own namespace. File permissions are
restricted at creation time where the platform supports them.

Updates acquire one application-level async lock and snapshot every mutable provider
before applying changes. Any validation or persistence error triggers a best-effort
rollback of all providers, preventing concurrent requests or a late ASR/TTS error from
leaving an earlier search/LLM mutation active. The resume client aliases the main
LLM only until its first dedicated configuration; that update constructs a separate
client, atomically swaps the service reference, and participates in rollback and
shutdown like every other transport. Audio settings also carry a persisted revision,
opaque ETag, and local effective-settings fingerprint. That fingerprint includes a
one-way digest of each configured API key so rotating only a credential still fences
old captures, but neither the raw key nor digest is returned by the API. Capture
registration binds to that generation, so a recording started under one ASR/omni
configuration cannot be finalized under another.

Each text-model request leases the transport and credential snapshot that was current
when it began. A base-URL hot switch publishes a new HTTP client synchronously but
does not close the retired client until its in-flight requests release their leases.
Capability-cache writes are generation-checked so an old endpoint cannot update the
new endpoint's structured-output capability.

Audio work follows the same lease model: live upload, ASR preview, mock
transcription, and background delivery analysis retain the effective ASR/omni
transport and generation until completion. A settings transaction can publish new
clients immediately but defers closing retired clients until all leases release; a
late background result is generation-checked before it can be attached to state.

## Client Runtime and Deployment Topologies

One dependency-free static application supports both launch modes. View modules call
the client created by `web/modules/api.js`; they never concatenate a backend origin
or construct authentication headers themselves. The transport reads its endpoint
contract from `web/modules/runtime.js`, attaches account and desktop credentials last,
links cancellation signals, and rejects a delayed response when the selected account
or session has changed.

The renderer treats account/session changes as data-boundary invalidations. It tracks
dirty editors explicitly, preserves only those dirty values in an in-memory recovery
snapshot, and restores them only after the same username reauthenticates and the
server confirms ownership of the captured session. Otherwise forms, consent flags,
modal datasets, audio playback, and secret fields are cleared. Visibility handling is
ownership-aware: hiding an active continuous listener (or pending continuous guard)
runs the normal pause/drain path; all other pending microphone starts simply recheck
visibility after asynchronous permission/authority boundaries and refuse activation.

### Standalone Browser Server

```text
Browser Web UI  -- same-origin HTTP -->  FastAPI  --> InterviewService
     |                                      |
     +-- bearer/session selection           +-- SQLite + local files
```

The normal browser launch serves HTML, CSS, and ES modules from FastAPI on the same
origin as the API. The runtime adapter can also consume an explicitly injected
HTTP(S) API origin for a controlled deployment. The browser stores the account
bearer token, selected role, and selected session ID in local storage. Registration
and login establish account credentials; business, settings, and Debug routes require
the account bearer token when authentication is enabled. Debug additionally rejects
non-loopback clients.

### Packaged Desktop Application

```text
Tauri WebView -- invoke --> Rust launcher -- private stdin --> Python sidecar
     |                         |                                  |
     +-- bundled Web assets    +-- random port/token              +-- FastAPI
     +---------------- authenticated loopback HTTP -------------------+
```

The desktop bundle loads the same Web assets directly in a Tauri WebView. The Rust
launcher generates a 256-bit bootstrap token, starts the native PyInstaller sidecar
with an OS application-data directory, and sends the token over the child's private
stdin. The Python process clears inherited database/settings/recording overrides,
binds one Uvicorn worker to a reserved random `127.0.0.1` port, and emits a single
versioned readiness record without the token. Rust accepts only protocol version 1,
desktop mode, and a non-zero `127.0.0.1` endpoint before returning the in-memory
runtime configuration to the WebView.

The browser-server preserves legacy storage defaults (`interview_os.db` and
`data/recordings` below its working directory, plus settings below
`~/.interview_os`) unless the operator configures overrides. The desktop sidecar
consolidates its database, settings, recordings, model/search metrics, Debug events,
and search cache below Tauri's per-user local application-data directory. Neither
topology supports multiple FastAPI workers for one data directory: session locks and
background-task ownership are process-local. Tauri also enforces one application
instance so two bundled sidecars do not share the same SQLite state.

## Desktop Lifecycle

Desktop readiness and shutdown are explicit protocols rather than timing guesses:

- The WebView waits for the validated readiness record for at most 90 seconds. A
  timeout atomically claims the `Starting -> Failed` transition and only that winner
  kills the child. A readiness record that loses this race cannot revive the runtime;
  any startup failure produces a local-service error, and normal app-exit cleanup
  owns any remaining child process.
- Once credentials have reached the WebView, an unexpected sidecar exit closes the
  application. The launcher does not auto-restart or let a new local process inherit
  an authenticated port.
- Closing first asks the renderer to stop permission races and suggestion streams,
  freeze microphones, reconcile an in-flight transition, flush queued audio, upload
  whole-session parts, and pause the live state. A failed drain cancels the close so
  the user can recover rather than silently losing audio; a 150-second Rust watchdog
  is only a final UI-hang escape.
- After the renderer acknowledges completion, Rust sends a versioned shutdown command
  over stdin. Uvicorn receives a graceful-shutdown request, the application cancels
  tracked background tasks and closes distinct transports before storage, and the
  launcher kills the child only after the bounded shutdown deadline. Repeated close
  requests cannot bypass the same cleanup operation.

Renderer readiness is scoped to a WebView document generation, not the desktop
process. A navigation/reload invalidates the former bridge acknowledgement before the
new document installs one. If navigation or a renderer crash destroys a document
while close is already pending, Rust marks the renderer phase complete and continues
native shutdown immediately instead of waiting for the 150-second watchdog.

## Desktop Security and Packaging Boundary

The process-lifetime bootstrap token is launcher isolation, not account authentication.
The sidecar requires it on `/health` and every non-OPTIONS `/api/*` request, including
login and registration; authenticated business routes additionally require the normal
account bearer. Static bundle assets do not carry this header. The Tauri Content
Security Policy limits scripts to bundled code and network/media access to the private
loopback service, and sidecar stderr is not forwarded into the renderer.

This boundary reduces accidental access and local port-rebinding risk, but it is not
TLS, an OS sandbox, or protection against malicious code already running as the same
user. CORS and loopback binding are defense-in-depth, not authorization. Browser-mode
tokens remain in local storage, desktop bootstrap credentials remain in renderer
memory for the process lifetime, and this layer does not add at-rest encryption.
On POSIX filesystems the recording root is normalized to `0700` and all newly written
or reconciled recording files to `0600`; equivalent privacy relies on native ACLs on
platforms without POSIX modes.

Packaging builds one native sidecar for the current OS and CPU and embeds it as a
Tauri external binary; it is not a universal or cross-compiled artifact. On macOS,
the release guard requires an installed Developer ID Application identity and
notarization credentials, and the release entitlement enables microphone input only.
The separate local-test configuration is deliberately ad-hoc and disables library
validation for the PyInstaller child; its guard refuses Apple release credentials and
that bundle must not be distributed. A successful local macOS build does not assert
that release notarization or native Windows/Linux installer, microphone, and signing
acceptance have been completed.

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

When persistent runtime storage is enabled, Debug creates its parent with mode `0700`
and its atomically replaced file with `0600` where supported. Persistence is
fail-open: serialization, disk-full, read-only, or replacement failures leave the
bounded redacted event in memory and never alter the business operation being
observed.

## Streaming Suggestions

Text and omni model adapters use HTTPX streaming contexts so upstream SSE bytes are
consumed before the completion finishes. The API relays JSON-encoded `append` and
`replace` SSE records, preserving embedded newlines, and persists the assembled
suggestion after the stream completes. A transport failure after partial output emits
a redacted deterministic replacement, so neither browser text nor persisted state
contains the internal error or incomplete suggestion. Streaming consumers use
`api.raw`, the same unified transport as JSON/blob/text calls, so account bearer and
desktop bootstrap headers, abort propagation, and stale account/session rejection do
not diverge from ordinary requests. Disconnecting closes the response generator and
upstream HTTP stream. Suggestion write-back rechecks that the live session is still
active, discarding completions that arrive after pause or finish.
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

Agents communicate via typed `Message` objects through an `AgentRuntime`. Each
owner/session pair has its own runtime and `InterviewState`; `InterviewService`
reloads that state from durable storage after a cache miss or uncertain commit.

## Key Design Decisions

- **Evidence-based evaluation**: every assessment is backed by Evidence records
- **Three-way fusion**: Candidate + Job + Interviewer → Strategy
- **State machine**: InterviewStage tracks where in the process we are
- **One client contract**: browser and desktop views share the same API modules
- **Manifest-led audio recovery**: persisted state decides which local files survive
