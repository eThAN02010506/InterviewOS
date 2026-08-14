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

### LAN access with microphone support

Browsers allow microphone capture only in a secure context. `localhost` is a
development exception, but `http://192.168.x.x:8000` is not: another device can
open the UI yet the browser will refuse recording permission. Run the HTTPS LAN
entry point instead (replace the address with this Mac's current LAN IPv4):

```bash
scripts/run_lan_https.sh 192.168.1.16
```

Then open `https://192.168.1.16:8443`. On each client device, install and trust
`data/tls/interview-os-local-ca.crt` once, then reopen the HTTPS page. The script
keeps that local CA stable and regenerates only the server certificate so DHCP
address changes can be handled by rerunning it with the new IP. All generated
certificates and private keys live under the git-ignored `data/` directory; never
copy the `.key` files to client devices. HTTP remains suitable for local text-only
development. A non-local HTTP page now explains that microphone access requires
HTTPS instead of reporting a generic browser failure.

Login is required before using the app: register a local account (username +
password, hashed with PBKDF2) and sign in. Session data is isolated per account —
an account only ever sees its own sessions, and a cross-account session lookup
returns 404 rather than leaking existence. Existing sessions created before
accounts existed are backfilled to a sentinel `local` owner; the first real account
registered after upgrade atomically claims those legacy sessions (`local` is a
reserved migration username and cannot be registered). Business, settings, and localhost Debug APIs all
require authentication; Debug session/event views remain scoped to the signed-in
account. Tests can explicitly
set `INTERVIEW_OS_REQUIRE_AUTH=0`; the production default is enabled.
Concurrent registration attempts for the same normalized username are resolved by
the database unique constraint and consistently returned as HTTP 409 conflicts.

The interviewer workspace includes a **Live Interview Copilot**. With explicit
consent, it processes typed or recorded interview turns and prepares the next
follow-up or main question. It is a decision-support surface: the interviewer
chooses, edits, skips, or postpones every suggested question. See the
[product requirements](docs/product_requirements.md) for scope, privacy rules,
delivery phases, and acceptance criteria.

Whole-session audio has one authoritative file per session. If the browser's
page-exit fallback replaces a normal WAV save with WebM or M4A, the persisted
filename controls downloads and superseded encodings are removed after the new
state is durable; download media types continue to match the stored encoding.

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
- Resume parsing can identify suspicious or incomplete claims. Once review items
  exist, downstream preparation Agents receive only confirmed or user-modified
  claims. A directly pasted resume with no review record is labelled as unverified
  candidate self-report and may guide preparation, but cannot become evaluation
  evidence. The UI labels confirmation as candidate/user confirmation rather than
  external truth verification, keeps reviewed items visible, and lets reviewers
  restore an item to unverified status after a mistaken action.
- A session is one candidate evidence boundary. Before any real answer, transcript,
  evidence, or recording exists, changing the resume invalidates every derived
  strategy, blueprint, mock plan, and report. After interview activity exists, the
  service refuses resume replacement or workflow regeneration and asks for a new
  session rather than silently deleting or mixing candidate data. Upload rejection
  happens before document parsing or optional LLM structuring, then is checked again
  before committing results so concurrent interview activity cannot be overwritten.
- JD quality matters. A title-only JD is treated as insufficient context, and the
  UI asks for responsibilities, requirements, and team background while separating
  explicit requirements from AI assumptions. With explicit public-research consent,
  a title-only input first triggers a source-filtered public JD search; source-bound
  snippets are supplied to the Job Agent before questions are generated. The stored
  user input remains the original title, every discovered requirement stays
  `inferred`, and source cards/warnings require confirmation. The same provenance
  restoration runs on workflow failure, so internal source-enriched prompt text is
  never retained as the user's JD. Without consent or a
  relevant source, the workflow stays generic. Inferred requirements can be
  confirmed, edited into explicit requirements, or removed before later Agents use
  them.
- Search conclusions are organized into fact cards with source URLs and status:
  verified, inferred, conflicting, or needs review. The UI groups cards by conflict,
  company, interviewer, past-employer, technology, and public-opinion categories.
  Each card shows source count, best source quality, provider fetch time,
  cache-hit status, filtering reason, and generation time. Users can confirm a
  card for later context, reject it, or reset it to inferred review state.
- The candidate's past employers can be researched too, but this lookup is strictly
  **company business background**, not candidate evidence. Queries target official
  About/product/business pages; current vacancies, Careers/Jobs pages, recruiting
  aggregators, and hiring terms are excluded. Downstream prompts may use these
  sources only to understand what the company does and must never infer the
  candidate's duties, skills, achievements, or role from them. Reviewed resumes
  expose only employer names present in confirmed/modified claims and still require
  the public-research consent gate. Direct pasted workflows without review data
  treat employer names as unverified self-report. The UI labels these cards
  "过往雇主业务背景（不代表候选人经历）".
- Secrets must not appear in SQLite, API responses, logs, or the Debug Console.
  Resume and transcript content necessarily lives in the owner-scoped local session
  database; it must not be copied into logs or Debug events.

The API now persists session state in SQLite. A minimal analysis flow is:

PDF and Word (`.docx`) resumes can be uploaded through
`POST /api/resumes/{session_id}/upload`. Files are parsed locally before the Agent
workflow runs. PDF text is extracted with pdfplumber (replacing pypdf), which keeps
structural line breaks and table rows instead of emitting pypdf's column-alignment
padding; docx tables are extracted row by row so multi-row tables do not collapse
into one long line. The response includes extraction warnings and human-review items
for education, employment, certifications, and measurable claims. Uploading a
resume never starts public web research automatically.

Resume parsing is rule-based by default, but an optional **AI structured
enhancement** is available: check the "AI 结构化增强" box on upload (or pass
`structure=llm`) and a local multimodal LLM (configured in Settings → resume_llm,
e.g. Qwen3-Omni on 8004) parses the extracted text into semantic sections
(education / employment / research / leadership / awards / skills). This fixes the
rule matcher's blind spots — a research description that mentions a university is
no longer mistaken for education, and awards are no longer classified as jobs. On
the real Mingyuan resume, LLM structuring produced 15 correctly-classified
sections where the rules produced 8 with several misclassifications. The rules
result is always kept as a fallback when the LLM is unavailable.

1. `POST /api/interviews/sessions`
2. `POST /api/analysis/resume` with the returned `session_id`
3. `POST /api/analysis/job` and `POST /api/analysis/interviewer`
4. `GET /api/interviews/sessions/{session_id}` to retrieve the accumulated state

For an end-to-end flow, call one of:

- `POST /api/autopilot/{session_id}/run` — automatically executes every authorized
  analysis and planning step. Candidate mode starts the AI-led interview, evaluates
  each answer, and generates the final dual-side report after the last answer.
  Autopilot pauses instead of fabricating candidate answers or real-world evidence.
  Explicit public-research consent is propagated to every nested company/interviewer
  workflow; without it, no search provider is called. The session's candidate display
  name survives document upload and model output that omits identity.

- `POST /api/workflows/candidate-prep` — candidate, job, company and optional
  interviewer analysis, followed by a structured strategy and mock interview plan.
- `POST /api/workflows/enterprise-design` — candidate, job and company analysis,
  followed by a structured multi-round interview blueprint.

Workflow progress is stored under `state.workflow`. Clients can poll the existing
session endpoint while a workflow request is running. Invalid structured LLM output
marks the workflow as `failed` and preserves the failing step and error message.
Structured calls use temperature zero and send the Pydantic JSON Schema through
OpenAI-compatible `response_format`. A provider that rejects this optional transport
hint is detected once and retried without it. When the configured model is `gpt-oss`, the
OpenAI-compatible request sends `chat_template_kwargs.reasoning_effort=low`, the
location llama.cpp's GPT-OSS Jinja template actually reads. This prevents the model
from exhausting its completion budget in hidden reasoning and returning empty JSON.
Transient local-model connection, timeout, HTTP 429/5xx, and malformed successful
responses are retried once in the transport adapter before structured-output retries
begin. A `200` without usable `choices.message.content` becomes a redacted model error
instead of escaping as `KeyError`. Provider exception and response text are never
returned to agents or written to logs.
Schema-invalid content is retried once with only the missing/invalid field locations
and validation types, never the raw model output. Debug events expose that sanitized
reason, attempt number, response length, and whether JSON Schema mode was requested.
If the configured model still returns invalid scoring JSON, the coach applies a
transparent deterministic rubric based on answer detail, concrete actions, structure,
results, and verified metrics. The UI identifies this as a rule score requiring human
review; it no longer assigns every failed response the same arbitrary 40 points.
Public fact cards are created only from substantive source snippets; image captions,
navigation text, and page titles remain visible as raw sources but are not promoted
to verified facts. Multi-entity person/company searches recognize either quoted
entity's matching official domain.

After candidate preparation, run an interactive mock interview:

1. `POST /api/mock-interviews/{session_id}/start`
2. Present the returned `current_question` (each question carries an
   `answer_framework` — a reference hint about which resume experience to tell,
   what structure to follow, and which signals to emphasize, generated per
   question by the model).
3. `POST /api/mock-interviews/{session_id}/answers` with its `question_id` and
   answer. A retry also sends `retry: true` plus the exact `retry_response_id`,
   so retrying a follow-up replaces that follow-up rather than its parent answer.
4. `POST /api/mock-interviews/{session_id}/next` advances to the next question
   (or offers an evidence-seeking follow-up when signals are missing; advancing
   may skip an unanswered question — the interviewer stays in control).
5. `POST /api/mock-interviews/{session_id}/previous` goes back to the previous
   question.
6. `POST /api/mock-interviews/{session_id}/finish` ends the interview and runs
   the evaluation.

Immediately after a main or follow-up answer is submitted, the question panel keeps
showing the exact `MockAnswerRecord.question` associated with the visible coaching
result. It switches to the next main question only after explicit navigation, so a
follow-up score is never displayed beside its parent question text.

The mock interview is **unlimited**: it never auto-completes after a fixed
question list. The question pool starts from the strategy's likely questions
plus job-competency templates, and a background refill keeps ~3 questions
cached — when fewer than 3 remain, 2 more are generated (grounded in the
resume and recent answers). The interview ends only when the interviewer
chooses 结束面试. During the interview 上一题 / 下一题 / 结束面试 stay
available so the interviewer can navigate freely. At a pool boundary, a unique
deterministic question is inserted immediately while the model refill continues,
so a slow, failed, or duplicate-only refill cannot leave an active interview
without a current question. An answered question can only be submitted again
through explicit retry; the service replaces its response and linked evidence
instead of silently creating duplicates. Replacing a main answer also invalidates
the follow-up responses and evidence derived from that superseded answer. Superseded
records move to an immutable attempt history only after the replacement has been
scored successfully, so provider failure cannot erase the current answer. When both a
main answer and follow-up exist, the UI exposes separate “重答主问题” and “重答当前追问”
actions so the backend's precise replacement behavior remains reachable. Final evaluation transitions through a
recoverable `evaluating` state: model failure preserves every answer and returns
the session to `active`, allowing the user to finish again. Process-local refill
flags and an interrupted `evaluating` state are restored to retryable values after
a process restart.

Mock questions now have an explicit quality contract. A main question targets one
competency and asks for a bounded, completed situation with the candidate's own
role, decision/action, constraints or trade-offs, observable result, and reflection.
Short or open-ended model output such as “talk about your experience” is rewritten
into that evidence-seeking form before it reaches the UI. Each question displays
an explicit, question-type-specific set of requirements later used by answer-coverage feedback. It also provides
two separate learning aids: a candidate-grounded selection/framework hint and a
complete, realistic teaching example. The complete example is explicitly fictional,
uses no resume company/project/metric, and demonstrates evidence density rather than
claiming to be the candidate's answer.

The candidate UI keeps the next action explicit throughout this lifecycle. The home
CTA routes to preparation, the active mock, or the growth report according to session
state. During model work, forms expose an accessible busy state and retain the user's
input on failure. Finishing a mock no longer clears the final coaching result: the
completion card summarizes answer count and average score, preserves the last answer's
feedback, and offers direct actions to open the report or create a separate practice
session. Primary, secondary, disabled, focus, and sticky navigation states share one
responsive hierarchy across desktop and mobile layouts.

Candidates can answer by voice instead of typing. While recording, bounded
cumulative audio snapshots are sent to `POST /api/live-interviews/{session_id}/
audio/preview`; the provisional text appears in the answer box but is never
persisted as transcript or evidence. Stopping the recording sends one stable WAV
to `POST /api/mock-interviews/{session_id}/transcribe`, keeps the original browser
recording available in an audio player, and lets the candidate edit the final
text before submitting. The server stages that recording under an opaque ID;
submitting the answer binds it to the response. An owner-checked audio endpoint
allows replay after refresh or navigation. Retry keeps the prior attempt and its
recording for side-by-side coaching; the user can explicitly delete either recording
without deleting its text or score. Unsubmitted staging files become cleanup
candidates after 24 hours, while active and archived response recordings are retained.
Uploads are limited to signature-validated WAV, WebM, and M4A files (25 MB maximum).
Recording, replay, provisional ASR, and delivery feedback are scoped to the session
where recording started. Switching sessions stops active capture, revokes browser
object URLs, clears coaching output, and prevents a late transcription response from
being applied to the newly selected candidate.

Every submitted answer also gets auditable spoken-answer analysis. The original
transcript remains evidence while a separate cleaned semantic draft removes
non-semantic fillers. The `evidence-v2` bilingual, cross-domain rubric classifies the
requested answer type,
extracts ordered steps with source excerpts, and checks coverage of the case,
method, execution, personal-decision, and result requirements. A downward-only
second pass caps model scores that exceed observable evidence and exposes every
adjustment plus the pre-calibration score in the UI. Input modality is explicit:
typed answers are not penalized for conversational transition words, while ASR/live
ASR answers may receive a structure cap only for strong fillers or repeated repairs.

The current question can be read aloud through the configurable OpenAI-compatible
TTS client (`8002` / Qwen3-TTS by default). The server only accepts the current
owned question ID; an optional response ID binds replay to the exact persisted main
or follow-up question. The browser aborts obsolete requests and rejects late audio,
so navigation cannot replace the visible question with stale speech. The endpoint
cannot be used as an arbitrary speech proxy.
After transcription, the configured `8004` omni model may analyze only changeable
delivery features: pace, pauses, fillers, volume stability, intonation, and clarity.
Transcription returns immediately with deterministic feedback; deeper audio analysis
runs in the background and the UI polls for the persisted result. Answer scoring shows
waiting stages, then prioritizes one actionable improvement; full rubric evidence is
collapsed on demand and retries show score/coverage deltas against the previous attempt.
It is forbidden from evaluating accent, personality, health, demographic traits,
or hire suitability. Provider output is screened again on the server; prohibited
inferences are discarded rather than displayed. If audio analysis fails, violates
that boundary, or exceeds 25 seconds, deterministic
duration/text indicators are returned. Delivery feedback is coaching-only and is
never added to competency evidence or the hiring recommendation.

Each answer receives validated 0–1 scores for four equally weighted, job-related
dimensions: role-relevant evidence, decision/professional depth, response structure,
and result/reflection. Their mean becomes evidence confidence for the question
competency. The service then creates a behavior-anchored card for every dimension:
the observable answer feature supporting the score and one concrete next action.
Model-generated `observed_signals` remain coaching hints only: persisted Evidence
always quotes the candidate answer and stays neutral until explicit human review.
Missing/partial question requirements become the primary ranked improvements, ahead
of generic dimension advice. The UI places “what this question asked / what your
answer actually said” outside the collapsed scoring details, quotes the matching
answer excerpt for every covered item, and explains exactly how to fill each gap.
The model does not generate a replacement candidate answer. It scores the submission
and returns at most three feedback/signal items; InterviewOS deterministically
reorganizes the candidate's exact wording into a question-specific rehearsal draft,
with explicit placeholders only where scenario, ownership, action, result, validation,
or reflection is absent. This prevents model-written names, meetings, tools, dates,
percentages, headcount, money, and outcomes from becoming candidate claims. Any legacy
model draft carrying unsupported numbers is discarded.
If no LLM is configured, or the optional per-question framework pass is incomplete,
every question receives a deterministic candidate-aware answer framework.

When evidence is available, `POST /api/evaluations/{session_id}` runs the final
evaluation and feedback workflow. Candidate UI presents strengths, improvements,
and an action plan; interviewer UI presents competency scores, evidence confidence,
signal gaps, and a hiring recommendation. Missing evidence returns `409` instead of
fabricating a conclusion.
Candidate-facing `overall` and `action_plan` fields are post-validated: hiring or
employment recommendations are replaced with a preparation-only evidence summary,
and interviewer commands such as “要求候选人…” are converted into direct candidate
practice actions. Hiring decisions remain exclusive to the interviewer workspace.
Interviewer recommendations use a locked evidence core: every competency score,
confidence, supporting signal, gap, risk, `overall_score`, and recommendation is rebuilt
from persisted `Evidence` records before fixed thresholds are applied. The evaluation
agent then makes one optional structured LLM call for the narrative layer only. For each
competency the model must cite numbered Evidence from that same competency and return an
assessment plus one next verification question; it cannot return scores, recommendations,
or new gaps. Competency-name and Evidence-reference validation is atomic, and invalid
output leaves the complete deterministic report intact. The UI labels whether the prose
is model-generated or deterministic. `strong_hire`
requires at least 0.85 score and 0.75 aggregate confidence, so a model cannot promote
weak evidence by returning matching competency names with invented high scores.
The answer-scoring model returns an untrusted `AnswerEvaluationDraft`; the service adds
`scoring_source` and `review_status` itself before persistence, so model output cannot
claim human-review provenance. An unreviewed
deterministic rule score (including migrated legacy records) makes the entire hiring
recommendation `insufficient_evidence`, regardless of the numeric average. Negative
interviewer notes require an exact persisted evidence signal explicitly classified as
negative by a human reviewer and include its Evidence ID. Missing descriptions remain
“pending verification” and are never inferred as negative from free text.
The candidate and interviewer UIs expose “人工复核评分”; the review endpoint updates the
four rubric scores and evidence polarity, marks provenance as `human/reviewed`, rejects
records without linked evidence, and invalidates stale final reports. A late background
AI score cannot overwrite that human decision.
The interviewer recommendation explanation is generated deterministically from the
final calibrated enum and score, so model prose cannot say `lean_no_hire` while the
decision header correctly says `insufficient_evidence`.

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

Runtime search and Local LLM settings are changed from the authenticated Settings
workspace in the main UI. Secrets are write-only and are never returned to the
browser. UI changes take effect immediately for existing sessions;
settings are stored in a local permission-restricted file and environment variables
can still provide startup defaults.

Each settings request is serialized and applied transactionally across search, main
LLM, ASR, TTS, live-audio, and resume-LLM clients. Validation or local persistence
failure restores the previous runtime configuration. The resume structuring model
initially reuses the main LLM, but its first dedicated UI update creates a separate
client so changing resume extraction cannot silently move the primary reasoning
endpoint.

Runtime secrets are not written to SQLite, returned by the API, or included in the
Debug Console. A platform keychain or external secret manager can replace the local
settings store later without changing the settings API.

## Debug Console

The UI includes a read-only operational console backed by `/api/debug`. It shows
bounded Agent lifecycle events, durations, workflow failures, session summaries,
and a fixed LLM connectivity probe. Token and cost totals cover both ordinary and
streaming text-model calls; when a streaming provider omits usage metadata, the
accepted prompt and assembled completion are estimated locally. The console:

- accepts requests only from localhost;
- stores at most `DEBUG_EVENT_CAPACITY` events (500 by default);
- filters by account, session, and level before applying the requested result limit,
  so another account's event volume cannot hide the current account's diagnostics;
- truncates message output and never exposes API keys;
- reports structured-output failures as safe field/type diagnostics (for example,
  `impact:missing`) without storing the model response, resume, or answer text;
- does not provide arbitrary Python, shell, SQL, or prompt execution.

Background failures expose only exception types in logs and events. During app
shutdown, tracked scoring and question-refill tasks are cancelled before their
shared transports close; distinct text, resume, ASR, and omni clients then close
exactly once so a dedicated resume-model connection is not leaked.

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
created. Boundary confidence is explainable with factors such as linked interviewer
question, consecutive candidate segments, answer length, ASR involvement, and
ending-language cues. Confirmed live records can also be re-evaluated with an
updated question or competency while preserving the same transcript trace.

A **silence-based continuous listening MVP** is also implemented. It reuses
the existing HTTP ASR upload endpoint instead of introducing a second protocol too
early. Instead of fixed 15s chunks, the browser runs a Web Audio VAD: it keeps
recording continuously and finalizes a whole utterance when the speaker goes quiet
for ~0.6s (silence, not speaker change, is the end-of-utterance signal — a
candidate can finish while the interviewer stays silent, and the utterance still
finalizes). Each finalized utterance uploads sequentially as one self-contained
audio chunk (one MediaRecorder session per utterance, so every upload decodes
cleanly), shows queue and upload status, and marks each chunk as `unknown` speaker
by default. A live "正在听取 (Xs)" indicator shows while speech is ongoing. This
keeps the interviewer in control: the user must correct speaker/text and
explicitly confirm candidate answers before any chunk can become hiring evidence.
For longer interviews, the live state now maintains a bounded rolling transcript
summary and drops exact repeated transcript chunks before they can grow the
conversation context. The next-question planner receives the rolling summary,
recent confirmed turns, the interview blueprint, and confirmed evidence instead of
the full raw transcript.
The competency map also provides deterministic coverage guidance: for each target
competency it shows evidence count, strongest confidence, priority, why the signal
is still weak or sufficient, and a sample question the interviewer can ask next.
The live workspace also tracks blueprint question usage, separating pending,
suggested, and adopted questions so the interviewer can avoid repeating planned
questions and see which parts of the interview design remain unused.
An interviewer-facing action card now sits above the live side rail. It chooses one
next step from the same server-side state, prioritizing speaker review, answer
boundary merge, evidence confirmation, question adoption, coverage-gap planning,
and final evaluation readiness in that order. This keeps the UI from becoming a
pile of independent widgets during a real interview. The card's CTAs are now
actionable: safe operations such as starting/resuming, planning the next question,
confirming the first pending evidence item, merging the highest-confidence boundary,
and generating evaluation can be triggered directly, while judgment-heavy steps
such as editing speakers or deciding between adopted/edited/skipped questions
scroll and spotlight the exact review area.

The live loop now scores evidence in the background instead of blocking the next
question. Confirming a candidate answer returns immediately and places a
placeholder `live_interview` evidence row, so coverage guidance and the action
card's evidence count stay accurate while the coach evaluates asynchronously
(`scoring_status` transitions `scoring` → `scored`, surfaced in the review queue).
Next-question planning fires automatically after an answer is confirmed (skipped
when a suggestion is still pending to avoid duplicate LLM calls), and the live view
polls every 3 seconds to refresh the suggestion card and scoring status. This
removes the manual "confirm evidence, then click generate" wait from the hot path.

Next-question suggestions can also be **streamed** (SSE): the "流式生成" button
calls `POST /api/live-interviews/{id}/suggestions/stream`, which streams the model
output token by token so the suggestion card fills in live instead of appearing
after the full generation. The text and omni adapters keep an HTTPX streaming
context open rather than buffering a normal `post()` response; SSE data is JSON
encoded so model newlines cannot corrupt event framing, and the browser includes
the current account's bearer token on the raw streaming request. Events distinguish
incremental `append` from a safe `replace`: if the model disconnects after partial
output, the UI and persisted suggestion replace it with a generic fallback instead
of exposing transport details or saving incomplete text. A completion arriving after
the live session is paused or ended is discarded instead of entering the review queue. Transcript
ingestion stays chunked (the LAN ASR has no
streaming endpoint), but a finished chunk immediately produces a streaming
suggestion. Verified with the real 8001 model: tokens arrive incrementally and the
completed suggestion lands in the review queue (~8s full, first token ~1s).

The live audio path is switchable between two modes (Settings → 实时音频处理):
- **ASR + text (default)**: the classic path — audio to the LAN ASR, transcript to
  the text LLM.
- **Audio direct (omni)**: candidate audio goes straight to an OpenAI-compatible
  multimodal model (e.g. Qwen3-Omni on 8004), skipping ASR entirely; the model
  hears the answer and returns a next-question suggestion. A settings panel lets
  you name the provider, point it at a base URL/model, and probe its capability
  (a real audio sample is sent to verify the endpoint returns a usable question).
  It intentionally does not create a transcript segment (the model understands
  the audio directly); the suggestion is injected straight into the review queue.

  The audio-direct mode also supports **automatic speaker separation**: the
  "对话模式" button records a whole dialog (interviewer + candidate, no manual
  speaker selection), sends it to the omni model, which returns per-utterance
  speaker labels; consecutive utterances from the same speaker are merged into
  one transcript segment so a long candidate answer becomes one confirmable
  evidence segment. Verified with a real two-voice Chinese dialog: the model
  correctly split interviewer vs candidate across 4 turns. Manual speaker
  correction remains available if the split is ever wrong.
  In continuous mode the browser also drives speaker separation: each silence-
  finalized utterance uploads with `mode=dialogue`, so the omni model auto-labels
  the speaker for every utterance (the same diarize path as the 对话模式 button).
  Continuous mode thus works in both live-audio modes — in `asr_text` chunks land
  as `unknown`-speaker "待确认" segments, in `audio_direct` they land with an
  auto-detected speaker.

  The live workspace can also **record the whole interview as one audio file**.
  With the consent checkbox confirmed, a separate recorder captures the entire
  session (independent of the per-utterance VAD path, which discards silence
  gaps) and, on ending the interview, uploads it once to
  `POST /api/live-interviews/{id}/audio/final`. The WAV is saved under
  `data/recordings/{session_id}.wav` (a per-session file, never shared across
  accounts) and can be downloaded back from the live panel. The file lives on
  this machine only; the recording is opt-in via the same explicit consent that
  gates transcription.

  The direct path's context is **focused and priority-ordered** for speed: it
  carries only the job/covered competencies (the anchor, always kept), the
  recent stable conversation, and the most recent live evidence. Advancement-only
  information (blueprint backlog, rolling summary, coverage guidance) is
  excluded so prefill stays small. If the focused blocks still exceed the cap,
  blocks are dropped lowest-priority first — never the tail, so the anchor is
  never lost. Measured with a real Chinese WAV in a 7-round-blueprint session,
  the focused direct path returned a grounded follow-up in ~3.5s (vs ~8.9s with
  the full-context version, and vs ~32s for ASR + 20B text).

The planner's own LLM call is also bounded to keep it fast on a local model. The
question map sent to the model contains only unused blueprint questions plus the
question the last suggestion referenced (not the full blueprint), rolling-summary
context is capped at 1500 chars, and the evidence list is limited to live evidence.
A 512-token `max_tokens` cap avoids wasted decode. On a representative 5-round
blueprint the planner context shrinks ~33% (≈6.8k → ≈4.5k chars), cutting the
dominant prefill cost for the 20B local model.

Structured-output generation is resilient to the model drifting off-schema:
`think_structured` requests the model's JSON Schema at temperature zero and retries
once with a compact field-level validation diagnostic (no raw output or extra
candidate context), then agents with deterministic fallbacks
take over. Enterprise interview design now builds a generic JD-based blueprint
from job competencies when the model returns empty rounds, so a resume-less
interview (JD only, common before the candidate's resume arrives) no longer fails
the whole workflow.

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
2. **Completed silence-based continuous listening MVP** — keep the existing HTTP ASR upload path;
   the browser runs a Web Audio VAD and finalizes one whole utterance per ~0.6s of
   silence (a separate MediaRecorder session per utterance so every upload is a
   clean self-contained audio file), marks chunks as `unknown` speaker by default in
   `asr_text` mode (auto-separated in `audio_direct`), and requires transcript
   review before evidence is created. This gives a practical bridge before true
   streaming.
3. **Completed bounded context MVP** — exact duplicate transcript chunks are dropped,
   older stable turns are summarized into bounded context, and next-question planning
   uses summary + recent turns + evidence rather than the full transcript.
4. **Continuous streaming** — partial transcript events, stronger answer-boundary
   detection, WebSocket reconnect/protocol-level deduplication, and optional speaker diarization.
5. **Evidence map hardening** — confirmed turns already become `live_interview`
   evidence, can be revoked, merged, re-evaluated, and turned into coverage
   guidance, question-usage tracking, explainable boundary confidence, and a
   server-driven next-action card with clickable interviewer CTAs. Evidence scoring
   now runs in the background so it never blocks the next-question suggestion, and
   planning is auto-triggered after an answer is confirmed.
6. **Hardening** — long-interview tests, deterministic fallbacks, latency budgets,
   redacted observability, and cost measurement. A long 30-turn e2e regression
   asserts the bounded-context + dedup behavior and the 5s planning budget with a
   stub model.

The live path must not invoke an LLM for every partial word. Stable transcript turns
feed a bounded context made from recent dialogue, a rolling summary, the interview
blueprint, and confirmed evidence. The target is transcript feedback within two
seconds and a next-question suggestion within five seconds after an answer ends.

The default LAN ASR integration is `http://192.168.1.97:8007` (Qwen3-ASR-1.7B),
behind a configurable provider adapter rather than hard-coded into the interview
workflow. Its `/health`, OpenAI-compatible `/v1/audio/transcriptions`, WAV input,
and JSON text response have been verified; a multi-sentence Chinese WAV transcribes
in ~5.1s with clean text. `http://192.168.1.97:8003` (MiMo-V2.5-ASR) remains a
verified alternative. Streaming support and concurrency limits remain part of
long-interview hardening. When ASR is unavailable, the live workspace retains
typed/pasted transcript input as the safe fallback.

`http://192.168.1.8:9001` was tested with the same multi-sentence WAV fixture. Its
OpenAI-compatible endpoint was faster, but returned only the first sentence, so it
is not the default evidence source until that truncation behavior is resolved.

### Real HTTP smoke tests

Two release-oriented scripts exercise the configured local service with synthetic
data:

```bash
.venv/bin/python scripts/e2e_real_workflow.py --base-url http://127.0.0.1:8000
.venv/bin/python scripts/e2e_live_action_card.py --base-url http://127.0.0.1:8000
```

`e2e_live_action_card.py` verifies the interviewer-side Live Copilot loop: start
with consent, detect a mergeable answer boundary, confirm live evidence, adopt the
auto-planned next question (planning now fires automatically after evidence is
confirmed, no manual step), collect enough evidence, finish the live session, and
generate a hiring evaluation. On the 2026-08-05 local run against the configured
8001 text model, the script completed in 86.307 seconds; action-card planning took
24.631 seconds and final evaluation took 40.886 seconds, down from 113.823 / 33.033 /
51.191 on the 2026-08-02 run before scoring was made async and the planner context
was slimmed. That historical final-evaluation timing predates the current locked-core
aggregation plus one bounded narrative call; feedback aggregation itself remains deterministic.
The loop is usable end to end; the remaining gap to the 5s
next-question target is the 20B model's own generation time, not code-path work.

The full audio loop (the interviewer's primary input path) was verified against the
real ASR and model on 2026-08-05: uploading a Chinese WAV transcribed in ~7.9s
(`source=asr`), confirming the transcribed segment became evidence, background
scoring marked it `scored`, and next-question planning auto-fired — all without
code changes. A browser-driven walkthrough of every page (candidate prep, mock
interview, improvement report; enterprise design, live copilot, evidence review;
settings, debug console) rendered with zero console errors; JD-inferred-requirement
confirmation and resume-claim confirmation both round-tripped to the server. The
entity-resolution dialog and fact cards are covered by API unit tests but were not
UI-tested this run (they need a live public-search hit to trigger).

A few real-use robustness fixes shipped after these runs: resume text is optional
in autopilot/workflow requests (a JD-only interview no longer 422s); structured
output retries the same prompt once before falling back; and both the interview
design and interview strategy agents build deterministic generic outputs when the
model returns empty results, so a resume-less design or a bad fusion response no
longer fails the whole workflow.

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
