"""State management for the interview lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator

from interview_os.core.evidence import Evidence

# 证据门槛：至少需要多少条证据与多少个胜任力，才能形成招聘建议
MIN_EVIDENCE_COUNT = 3
MIN_COMPETENCY_COVERAGE = 2
CROSS_VALIDATION_EVIDENCE_COUNT = 2  # 单个胜任力需要多少条独立证据才算交叉验证

# 实时面试上下文边界
LIVE_RECENT_SEGMENT_WINDOW = 12  # 规划/去重/摘要保留的最近稳定片段窗口
LIVE_SUMMARY_CHAR_LIMIT = 2800  # 滚动摘要的截断字符上限

# 下一问题规划器上下文边界（压缩 prefill）
PLANNER_QUESTION_MAP_MAX = 12  # 规划器一次最多携带的未用蓝图题数量
PLANNER_SUMMARY_CHAR_LIMIT = 1500  # 规划器看到的滚动摘要截断字符上限
PLANNER_MAX_TOKENS = 512  # 规划器结构化输出的 max_tokens，避免 decode 阶段浪费

# 音频直连（全模态模型）上下文边界
OMNI_CONTEXT_CHAR_LIMIT = 2500  # 直连模型一次携带的完整上下文截断字符上限

# 模拟面试题目池/补题缓存
MOCK_POOL_TARGET = 5  # 初始题目池目标数量
MOCK_CACHE_MIN = 3  # 待答低于此数触发后台补题
MOCK_REFILL_BATCH = 2  # 每次补题生成的问题数

# 覆盖引导
COVERAGE_GUIDANCE_MAX_ITEMS = 8
WEAK_SIGNAL_THRESHOLD = 0.65  # 证据最强置信度低于此值判为信号偏弱

# 行动卡
ACTION_CARD_SOURCE_REFS_MAX = 8
QUESTION_USAGE_MAX_ITEMS = 40
ANSWER_BOUNDARY_SUGGESTIONS_MAX = 5
MIN_ANSWER_BOUNDARY_SEGMENTS = 2  # 合并为一个回答边界建议所需的最少连续候选人片段数

# 回答边界置信度打分系数
BOUNDARY_BASE_SCORE = 0.45
BOUNDARY_MIN_SCORE = 0.2
BOUNDARY_MAX_SCORE = 0.92
BOUNDARY_MERGE_THRESHOLD = 0.65  # 行动卡触发"合并连续回答"的置信度门槛


class InterviewStage(str, Enum):
    NOT_STARTED = "not_started"
    HR_SCREEN = "hr_screen"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"
    CULTURE_FIT = "culture_fit"
    WRAP_UP = "wrap_up"
    COMPLETED = "completed"


class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AutopilotStatus(str, Enum):
    OFF = "off"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"


class AutopilotState(BaseModel):
    enabled: bool = False
    status: AutopilotStatus = AutopilotStatus.OFF
    phase: str = ""
    completed_actions: list[str] = Field(default_factory=list)
    pause_reason: str = ""
    authorized_public_research: bool = False
    started_at: datetime | None = None
    updated_at: datetime | None = None


class WorkflowProgress(BaseModel):
    name: str = ""
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    error: str = ""


class CandidateProfile(BaseModel):
    name: str = ""
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unique_advantages: list[str] = Field(default_factory=list)
    raw_resume_text: str = ""


class ResumeIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResumeClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    CONFIRMED = "confirmed"
    MODIFIED = "modified"
    NEEDS_DOCUMENTS = "needs_documents"
    DISPUTED = "disputed"
    IGNORED = "ignored"


class ResumeFileMetadata(BaseModel):
    filename: str = ""
    file_type: str = ""
    size_bytes: int = 0
    page_count: int = 0
    character_count: int = 0


class ResumeValidationIssue(BaseModel):
    code: str
    severity: ResumeIssueSeverity
    message: str
    field: str = ""


class ResumeClaim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: str
    statement: str
    original_statement: str = ""
    status: ResumeClaimStatus = ResumeClaimStatus.UNVERIFIED
    verification_method: str = "candidate_confirmation"
    note: str = ""


class ResumeStructuredSection(BaseModel):
    """One structured resume entry from LLM-based parsing.

    LLM structuring classifies entries into semantic sections (education,
    employment, research, leadership, awards, skills) instead of relying on
    keyword matching, so a research description that mentions a university is
    not mistaken for education.
    """

    category: str
    institution: str = ""
    title: str = ""
    date_range: str = ""
    description: str = ""


class ResumeReview(BaseModel):
    metadata: ResumeFileMetadata = Field(default_factory=ResumeFileMetadata)
    issues: list[ResumeValidationIssue] = Field(default_factory=list)
    claims: list[ResumeClaim] = Field(default_factory=list)
    structured: list[ResumeStructuredSection] = Field(default_factory=list)
    structured_by: str = "rules"  # "rules" | "llm"
    reviewed_at: datetime | None = None


class JobDescription(BaseModel):
    title: str = ""
    raw_description: str = ""
    department: str = ""
    level: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)


class RequirementOrigin(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class JobRequirement(BaseModel):
    text: str
    origin: RequirementOrigin


class JobDescriptionReview(BaseModel):
    is_title_only: bool = False
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_sections: list[str] = Field(default_factory=list)
    requirements: list[JobRequirement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResolutionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EntityResolution(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    entity_type: str
    input_name: str
    proposed_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    source_urls: list[str] = Field(default_factory=list)
    status: ResolutionStatus = ResolutionStatus.PENDING
    resolved_at: datetime | None = None


class FactStatus(str, Enum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    CONFLICT = "conflict"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FactCard(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: str
    subject: str
    claim: str
    status: FactStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_urls: list[str] = Field(default_factory=list)
    source_quality: str = "unrated"
    source_count: int = 0
    source_fetched_at: datetime | None = None
    source_filter_reason: str = ""
    cache_hit: bool = False
    note: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None


class CompanyInfo(BaseModel):
    name: str = ""
    industry: str = ""
    stage: str = ""
    technology_stack: list[str] = Field(default_factory=list)
    culture: str = ""
    preferences: list[str] = Field(default_factory=list)
    dna: str = ""
    public_sources: list[dict[str, Any]] = Field(default_factory=list)
    public_research_status: str = "not_requested"


class InterviewerProfile(BaseModel):
    name: str = ""
    position: str = ""
    company: str = ""
    education: list[dict[str, Any]] = Field(default_factory=list)
    career_history: list[dict[str, Any]] = Field(default_factory=list)
    career_pattern: str = ""
    technical_focus: list[str] = Field(default_factory=list)
    communication_style: str = ""
    likely_preferences: list[str] = Field(default_factory=list)
    public_expressions: list[dict[str, Any]] = Field(default_factory=list)
    public_research_status: str = "not_requested"


class InterviewStrategy(BaseModel):
    summary: str = ""
    key_risks: list[str] = Field(default_factory=list)
    answer_framework: list[str] = Field(default_factory=list)
    topics_to_emphasize: list[str] = Field(default_factory=list)
    topics_to_avoid: list[str] = Field(default_factory=list)
    likely_questions: list[str] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question: str = ""
    competency: str = ""
    rationale: str = ""
    strong_signals: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    answer_framework: str = ""
    source: str = "initial"  # initial | likely | competency | refill


class InterviewRound(BaseModel):
    name: str = ""
    goal: str = ""
    questions: list[InterviewQuestion] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)


class InterviewBlueprint(BaseModel):
    position: str = ""
    rounds: list[InterviewRound] = Field(default_factory=list)


class MockInterviewPlan(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)


class AnswerEvaluation(BaseModel):
    content: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("content_score", "content"),
        serialization_alias="content",
    )
    technical_depth: float = Field(ge=0.0, le=1.0)
    structure: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    improved_answer: str = ""
    observed_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)

    @field_validator("content", "technical_depth", "structure", "impact", mode="before")
    @classmethod
    def normalize_percentage_score(cls, value: Any) -> Any:
        """Accept the common local-model 0-10/0-100 score convention."""
        if isinstance(value, (int, float)) and 1 < value <= 10:
            return value / 10
        if isinstance(value, (int, float)) and 10 < value <= 100:
            return value / 100
        return value

    def overall_score(self) -> float:
        return (self.content + self.technical_depth + self.structure + self.impact) / 4


class MockAnswerRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question_id: UUID
    question: str
    competency: str
    answer: str
    evaluation: AnswerEvaluation
    is_follow_up: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MockSessionStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"


class MockInterviewSession(BaseModel):
    status: MockSessionStatus = MockSessionStatus.IDLE
    current_question_index: int = 0
    responses: list[MockAnswerRecord] = Field(default_factory=list)
    pending_follow_up: str = ""
    pending_parent_question_id: UUID | None = None
    refill_in_flight: bool = False
    started_at: datetime | None = None
    completed_at: datetime | None = None


class HiringRecommendation(str, Enum):
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    LEAN_HIRE = "lean_hire"
    LEAN_NO_HIRE = "lean_no_hire"
    NO_HIRE = "no_hire"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CompetencyEvaluation(BaseModel):
    competency: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    competencies: list[CompetencyEvaluation] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: HiringRecommendation = HiringRecommendation.INSUFFICIENT_EVIDENCE
    summary: str = ""
    risks: list[str] = Field(default_factory=list)
    finalized_at: datetime | None = None


class FeedbackReport(BaseModel):
    overall: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)
    interviewer_notes: list[str] = Field(default_factory=list)
    recommendation_reasoning: str = ""


class LiveInterviewRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str
    competency: str
    evaluation: AnswerEvaluation = Field(
        default_factory=lambda: AnswerEvaluation(
            content=0.0, technical_depth=0.0, structure=0.0, impact=0.0
        )
    )
    source: str = "interview_transcript"
    transcript_segment_ids: list[UUID] = Field(default_factory=list)
    scoring_status: str = "pending"  # pending | scoring | scored | failed
    scoring_error: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LiveInterviewStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class TranscriptSpeaker(str, Enum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class TranscriptSegment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sequence: int = Field(ge=1)
    speaker: TranscriptSpeaker = TranscriptSpeaker.UNKNOWN
    text: str = Field(min_length=1, max_length=12000)
    stable: bool = True
    confirmed: bool = True
    source: str = "manual"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuestionSuggestionStatus(str, Enum):
    PENDING = "pending"
    ADOPTED = "adopted"
    EDITED = "edited"
    SKIPPED = "skipped"


class QuestionSuggestionType(str, Enum):
    FOLLOW_UP = "follow_up"
    NEXT_MAIN = "next_main"
    CLARIFY = "clarify"
    WRAP_UP = "wrap_up"


class QuestionSuggestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    suggested_question: str = Field(min_length=1, max_length=4000)
    question_type: QuestionSuggestionType = QuestionSuggestionType.FOLLOW_UP
    competency: str = Field(default="综合能力", min_length=1, max_length=200)
    rationale: str = Field(default="", max_length=2000)
    evidence_gap: str = Field(default="", max_length=2000)
    expected_signals: list[str] = Field(default_factory=list, max_length=8)
    source_question_id: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list, max_length=2)
    status: QuestionSuggestionStatus = QuestionSuggestionStatus.PENDING
    final_question: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("question_type", mode="before")
    @classmethod
    def normalize_legacy_question_type(cls, value: Any) -> Any:
        return QuestionSuggestionType.NEXT_MAIN if value == "main" else value


class LiveAnswerBoundarySuggestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question_segment_id: UUID | None = None
    answer_segment_ids: list[UUID] = Field(default_factory=list, min_length=2, max_length=20)
    suggested_competency: str = Field(default="综合能力", min_length=1, max_length=200)
    reason: str = Field(default="", max_length=1000)
    confidence_factors: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LiveCoverageGuidance(BaseModel):
    competency: str
    evidence_count: int = Field(default=0, ge=0)
    strongest_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: str = "medium"
    reason: str = ""
    suggested_question_type: str = ""
    sample_question: str = ""


class LiveQuestionUsage(BaseModel):
    question_id: str
    round_name: str = ""
    question: str = ""
    competency: str = ""
    status: str = "pending"
    suggested_count: int = 0
    last_suggestion_status: str = ""
    last_decided_at: datetime | None = None


class LiveActionCard(BaseModel):
    action_type: str = "start"
    priority: str = "medium"
    title: str = "开始实时面试"
    detail: str = "确认候选人知情同意后开始监听。"
    primary_cta: str = "开始实时会话"
    secondary_cta: str = ""
    evidence_status: str = ""
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LiveInterviewSession(BaseModel):
    status: LiveInterviewStatus = LiveInterviewStatus.IDLE
    consent_confirmed: bool = False
    current_speaker: TranscriptSpeaker = TranscriptSpeaker.UNKNOWN
    segments: list[TranscriptSegment] = Field(default_factory=list)
    suggestions: list[QuestionSuggestion] = Field(default_factory=list)
    answer_boundary_suggestions: list[LiveAnswerBoundarySuggestion] = Field(default_factory=list)
    coverage_guidance: list[LiveCoverageGuidance] = Field(default_factory=list)
    question_usage: list[LiveQuestionUsage] = Field(default_factory=list)
    action_card: LiveActionCard = Field(default_factory=LiveActionCard)
    rolling_summary: str = ""
    summarized_until_sequence: int = 0
    duplicate_segments_dropped: int = 0
    last_duplicate_reason: str = ""
    used_question_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    audio_file: str = ""
    audio_size_bytes: int = 0
    audio_saved_at: datetime | None = None


def _structured_employment(state: InterviewState) -> list[str]:
    """Render structured employment sections for the evidence context."""
    return [
        f"- {s.institution} | {s.title} | {s.date_range} | {s.description}"
        for s in state.resume_review.structured
        if s.category == "employment" and (s.institution or s.title)
    ]


class InterviewState(BaseModel):
    candidate: CandidateProfile = Field(default_factory=CandidateProfile)
    resume_review: ResumeReview = Field(default_factory=ResumeReview)
    job: JobDescription = Field(default_factory=JobDescription)
    job_review: JobDescriptionReview = Field(default_factory=JobDescriptionReview)
    company: CompanyInfo = Field(default_factory=CompanyInfo)
    interviewer: InterviewerProfile = Field(default_factory=InterviewerProfile)
    entity_resolutions: list[EntityResolution] = Field(default_factory=list)
    fact_cards: list[FactCard] = Field(default_factory=list)
    past_employer_sources: list[dict[str, Any]] = Field(default_factory=list)
    past_employer_research_status: str = "not_requested"
    past_employer_block: str = ""
    strategy: InterviewStrategy = Field(default_factory=InterviewStrategy)
    blueprint: InterviewBlueprint = Field(default_factory=InterviewBlueprint)
    mock_interview: MockInterviewPlan = Field(default_factory=MockInterviewPlan)
    mock_session: MockInterviewSession = Field(default_factory=MockInterviewSession)
    evaluation: EvaluationReport = Field(default_factory=EvaluationReport)
    feedback: FeedbackReport = Field(default_factory=FeedbackReport)
    live_interview_records: list[LiveInterviewRecord] = Field(default_factory=list)
    live_interview: LiveInterviewSession = Field(default_factory=LiveInterviewSession)
    workflow: WorkflowProgress = Field(default_factory=WorkflowProgress)
    autopilot: AutopilotState = Field(default_factory=AutopilotState)
    current_stage: InterviewStage = InterviewStage.NOT_STARTED
    evaluated_competencies: dict[str, float] = Field(default_factory=dict)
    missing_signals: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    next_action: str = ""

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        if ev.competency in self.evaluated_competencies:
            old = self.evaluated_competencies[ev.competency]
            self.evaluated_competencies[ev.competency] = old * 0.6 + ev.confidence * 0.4
        else:
            self.evaluated_competencies[ev.competency] = ev.confidence

    def mark_evaluated(self, competency: str, score: float) -> None:
        self.evaluated_competencies[competency] = score
        if competency in self.missing_signals:
            self.missing_signals.remove(competency)

    def is_complete(self) -> bool:
        return self.current_stage == InterviewStage.COMPLETED

    def confirmed_resume_facts(self) -> list[str]:
        return [
            claim.statement
            for claim in self.resume_review.claims
            if claim.status in {ResumeClaimStatus.CONFIRMED, ResumeClaimStatus.MODIFIED}
        ]

    def candidate_evidence_context(self, *, structure_required: bool = False) -> str:
        """Build the candidate's evidence context for generation agents.

        The full raw resume is always included (it is the authoritative history),
        so a sparse or unconfirmed claim list can never starve the agents of
        recent employers. Confirmed claims are prepended with a priority label;
        ``structure_required`` additionally includes the structured employment
        sections when present (cleaner recency ordering from the LLM pass).
        """
        blocks: list[str] = []
        confirmed = self.confirmed_resume_facts()
        if confirmed:
            blocks.append("已确认的简历事实（最高优先级）：\n" + "\n".join(confirmed))
        if structure_required:
            structured = _structured_employment(self)
            if structured:
                blocks.append("结构化工作经历（按最近优先）：\n" + "\n".join(structured))
        if self.candidate.raw_resume_text.strip():
            blocks.append("完整简历原文（权威履历）：\n" + self.candidate.raw_resume_text.strip())
        return "\n\n".join(blocks)

    def enforce_evaluation_evidence_floor(self) -> None:
        evidence_competencies = {
            item.competency for item in self.evidence if item.competency.strip()
        }
        if len(self.evidence) >= MIN_EVIDENCE_COUNT and len(evidence_competencies) >= MIN_COMPETENCY_COVERAGE:
            return
        self.evaluation.recommendation = HiringRecommendation.INSUFFICIENT_EVIDENCE
        warning = (
            f"至少需要 {MIN_EVIDENCE_COUNT} 条证据并覆盖 "
            f"{MIN_COMPETENCY_COVERAGE} 个胜任力，才能形成招聘建议"
        )
        if warning not in self.evaluation.risks:
            self.evaluation.risks.append(warning)
        if self.evaluation.summary and not self.evaluation.summary.startswith("证据门槛未满足"):
            self.evaluation.summary = f"证据门槛未满足。{self.evaluation.summary}"
        if self.feedback.overall:
            self.feedback.overall = "当前证据不足，单题表现仅供参考，不能形成录用结论。"
        if self.feedback.recommendation_reasoning:
            self.feedback.recommendation_reasoning = (
                f"证据门槛未满足：至少需要 {MIN_EVIDENCE_COUNT} 条证据并覆盖 "
                f"{MIN_COMPETENCY_COVERAGE} 个胜任力；"
                "当前不能给出录用或不录用建议。"
            )
        note = "证据不足不是负面证据，需要继续采集独立回答"
        self.feedback.interviewer_notes = [
            (
                f"当前仅有 {len(self.evidence)} 条证据，覆盖 "
                f"{len(evidence_competencies)} 个胜任力；未达到招聘决策门槛。"
            ),
            note,
        ]
    def summary(self) -> str:
        return (
            f"Stage: {self.current_stage.value} | "
            f"Evaluated: {list(self.evaluated_competencies.keys())} | "
            f"Missing: {self.missing_signals} | "
            f"Next: {self.next_action}"
        )
