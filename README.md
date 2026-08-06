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
created. Boundary confidence is explainable with factors such as linked interviewer
question, consecutive candidate segments, answer length, ASR involvement, and
ending-language cues. Confirmed live records can also be re-evaluated with an
updated question or competency while preserving the same transcript trace.

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
`think_structured` retries the same prompt once on invalid output (no extra
context, zero cost on the happy path), then agents with deterministic fallbacks
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

The default LAN ASR integration is `http://192.168.1.97:8003` (MiMo-V2.5-ASR),
behind a configurable provider adapter rather than hard-coded into the interview
workflow. Its `/health`, OpenAI-compatible `/v1/audio/transcriptions`, WAV input,
and JSON text response have been verified. Streaming support and concurrency limits
remain part of long-interview hardening. When ASR is unavailable, the live workspace
retains typed/pasted transcript input as the safe fallback.

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
was slimmed. The loop is usable end to end; the remaining gap to the 5s
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
