"""Resume, job, company, interviewer, and preparation workflows."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from interview_os.core.message import Message
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    AutopilotState,
    AutopilotStatus,
    CandidateProfile,
    EvaluationReport,
    FeedbackReport,
    InterviewBlueprint,
    InterviewerProfile,
    InterviewStage,
    InterviewState,
    InterviewStrategy,
    LiveInterviewSession,
    MockInterviewPlan,
    MockInterviewSession,
    RequirementOrigin,
    ResumeClaimStatus,
    ResumeReview,
    WorkflowProgress,
)
from interview_os.services.intelligence_service import (
    review_job_description,
)
from interview_os.services.resume_llm import structure_resume_with_llm
from interview_os.services.resume_ocr import ResumeOCRError, ocr_scanned_pdf
from interview_os.services.resume_service import (
    ResumeProcessingError,
    ScannedPDFError,
)
from interview_os.tools.web_search import (
    filter_employer_business_results,
    filter_entity_results,
    format_employer_business_context,
    format_search_results,
    merge_search_results,
)

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    CandidateSessionStateError,
    ResumeReviewStateError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class PreparationServiceMixin(InterviewServiceMixin):
    """Resume, job, company, interviewer, and preparation workflows."""

    async def analyze_resume(self, session_id: str, text: str) -> Message:
        runtime = await self._get_runtime(session_id)
        await self._prepare_resume_transition(session_id, runtime, text)
        return await self._run(session_id, "candidate_agent", text)

    async def upload_resume(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        *,
        structure: str = "rules",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        # Reject an unsafe replacement before parsing the document or sending
        # its text to an optional structuring model. The second check below is
        # still required because interview activity may arrive while parsing.
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
        # Parsing and optional LLM structuring are slow and do not touch session
        # state, so keep them outside the per-session mutation lock.
        structuring_client = self.resume_llm_client or self.llm_client
        try:
            text, review = await asyncio.to_thread(self.resume_processor.process, filename, content)
        except ScannedPDFError:
            try:
                ocr_text, pages, ocr_engine = await ocr_scanned_pdf(content, structuring_client)
            except ResumeOCRError as exc:
                self._record_debug(
                    "resume_ocr_failed",
                    session_id,
                    detail=f"{type(exc).__name__}; file_type=pdf",
                )
                raise ResumeProcessingError(
                    "扫描版 PDF 自动 OCR 失败，请确认 resume_llm 支持图片或上传可复制文字的文件"
                ) from None
            text, review = self.resume_processor.process_ocr_text(
                filename, content, ocr_text, pages
            )
            self._record_debug(
                "resume_ocr_completed",
                session_id,
                detail=f"pages={pages}; chars={len(text)}; provider={ocr_engine}",
            )
        if structure == "llm" and structuring_client is not None:
            sections = await structure_resume_with_llm(text, structuring_client)
            if sections:
                review.structured = sections
                review.structured_by = "llm"
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            candidate_name = runtime.state.candidate.name
            self._reset_candidate_outputs(runtime.state)
            # A newly uploaded document replaces all candidate-derived state.
            # Keeping the previous parsed profile would mix two resumes before
            # the candidate agent has a chance to analyze the new document.
            runtime.state.candidate = CandidateProfile(
                name=candidate_name,
                raw_resume_text=text,
            )
            runtime.state.resume_review = review
            runtime.state.past_employer_sources = []
            runtime.state.past_employer_research_status = "not_requested"
            runtime.state.past_employer_block = ""
            runtime.state.next_action = "Review resume checks, then continue the interview workflow"
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_processed",
                session_id,
                detail=f"{review.metadata.file_type}, {review.metadata.character_count} chars, "
                f"{len(review.issues)} issues, {len(review.claims)} claims, "
                f"structured={review.structured_by}",
            )
            return runtime.state

    @staticmethod
    def _has_substantive_interview_activity(state: InterviewState) -> bool:
        """Whether resetting candidate inputs would discard real interview data."""
        return bool(
            state.mock_session.responses
            or state.mock_session.answer_draft
            or state.live_interview.segments
            or state.live_interview_records
            or state.evidence
            or state.live_interview.audio_file
        )

    @classmethod
    def _assert_candidate_reset_allowed(cls, state: InterviewState) -> None:
        if cls._has_substantive_interview_activity(state):
            raise CandidateSessionStateError(
                "当前会话已有面试回答、转写、证据或录音；请新建会话后再更换简历或重新生成方案"
            )

    @staticmethod
    def _reset_candidate_outputs(state: InterviewState) -> None:
        """Invalidate every artifact derived from candidate input before regeneration."""
        state.entity_resolutions = []
        state.fact_cards = []
        state.past_employer_sources = []
        state.past_employer_research_status = "not_requested"
        state.past_employer_block = ""
        state.strategy = InterviewStrategy()
        state.blueprint = InterviewBlueprint()
        state.mock_interview = MockInterviewPlan()
        state.mock_session = MockInterviewSession()
        state.evaluation = EvaluationReport()
        state.feedback = FeedbackReport()
        state.live_interview_records = []
        state.live_interview = LiveInterviewSession()
        state.workflow = WorkflowProgress()
        state.autopilot = AutopilotState()
        state.evidence = []
        state.evaluated_competencies = {}
        state.missing_signals = []
        state.conversation_history = []
        state.current_stage = InterviewStage.NOT_STARTED
        state.next_action = "Regenerate candidate-dependent interview artifacts"

    @staticmethod
    def _invalidate_final_reports(state: InterviewState, next_action: str) -> bool:
        """Clear derived decisions when their underlying evidence changes.

        This is intentionally a no-op before any final report exists, so ordinary
        answer collection does not disturb the active interview stage. Once an
        evaluation has been produced, every evidence mutation returns the state
        to a retryable wrap-up phase and also reopens completed automation state.
        """
        has_derived_report = bool(
            state.evaluation.finalized_at
            or state.evaluation.competencies
            or state.feedback.overall
            or state.workflow.name == "evaluation"
            or state.current_stage == InterviewStage.COMPLETED
        )
        if not has_derived_report:
            return False
        state.evaluation = EvaluationReport()
        state.feedback = FeedbackReport()
        state.evaluated_competencies = {}
        state.current_stage = InterviewStage.WRAP_UP
        if state.workflow.name == "evaluation":
            state.workflow = WorkflowProgress()
        if state.autopilot.enabled and state.autopilot.status == AutopilotStatus.COMPLETED:
            state.autopilot.status = AutopilotStatus.WAITING_FOR_INPUT
            state.autopilot.phase = "evaluation_review"
            state.autopilot.pause_reason = "Evidence changed after final evaluation"
            state.autopilot.completed_actions = [
                action
                for action in state.autopilot.completed_actions
                if action not in {"final_evaluation", "feedback_generation"}
            ]
            state.autopilot.updated_at = datetime.now(timezone.utc)
        state.next_action = next_action
        return True

    async def _prepare_resume_transition(
        self, session_id: str, runtime: AgentRuntime, resume_text: str
    ) -> None:
        """Safely invalidate stale derived state before a workflow consumes a resume."""
        async with self._lock_for(session_id):
            self._assert_candidate_reset_allowed(runtime.state)
            candidate_name = runtime.state.candidate.name
            same_resume = runtime.state.candidate.raw_resume_text.strip() == resume_text.strip()
            self._reset_candidate_outputs(runtime.state)
            if not same_resume:
                runtime.state.candidate = CandidateProfile(
                    name=candidate_name,
                    raw_resume_text=resume_text,
                )
                runtime.state.resume_review = ResumeReview()
            else:
                runtime.state.candidate.raw_resume_text = resume_text
            await self._persist(session_id, runtime.state)

    async def update_resume_claim(
        self,
        session_id: str,
        claim_id: UUID,
        status: ResumeClaimStatus,
        note: str = "",
        statement: str | None = None,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            claim = next(
                (item for item in runtime.state.resume_review.claims if item.id == claim_id), None
            )
            if claim is None:
                raise ResumeReviewStateError("Resume claim not found")
            claim.status = status
            claim.note = note.strip()[:500]
            if statement is not None:
                revised = statement.strip()[:2000]
                if not revised:
                    raise ResumeReviewStateError("Resume claim statement cannot be empty")
                claim.original_statement = claim.original_statement or claim.statement
                claim.statement = revised
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "resume_claim_updated", session_id, detail=f"{claim.category}: {status}"
            )
            return runtime.state

    async def analyze_job(self, session_id: str, text: str) -> Message:
        return await self._run(session_id, "job_agent", text)

    async def update_job_requirement(
        self,
        session_id: str,
        index: int,
        *,
        action: str,
        text: str = "",
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        clean_text = text.strip()
        async with self._lock_for(session_id):
            requirements = runtime.state.job_review.requirements
            if index < 0 or index >= len(requirements):
                raise ResumeReviewStateError("Job requirement not found")
            if action == "delete":
                requirements.pop(index)
            elif action == "confirm":
                requirements[index].origin = RequirementOrigin.EXPLICIT
            elif action == "edit":
                if not clean_text:
                    raise ResumeReviewStateError("Job requirement text cannot be empty")
                requirements[index].text = clean_text
                requirements[index].origin = RequirementOrigin.EXPLICIT
            else:
                raise ResumeReviewStateError("Unsupported job requirement action")
            explicit_count = sum(
                1 for item in requirements if item.origin == RequirementOrigin.EXPLICIT
            )
            if explicit_count:
                runtime.state.job_review.is_title_only = False
                runtime.state.job_review.completeness_score = max(
                    runtime.state.job_review.completeness_score,
                    min(1.0, 0.45 + explicit_count * 0.05),
                )
            await self._persist(session_id, runtime.state)
            self._record_debug(
                "job_requirement_updated",
                session_id,
                detail=f"index={index}; action={action}",
            )
            return runtime.state

    async def analyze_company(self, session_id: str, name: str, context: str = "") -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.company.name = name
            runtime.state.company.context = context.strip()
            message = await runtime.run("company_agent", context)
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            return message

    def _recent_employers(self, state: InterviewState, limit: int = 2) -> list[str]:
        """Pick the past employers worth researching.

        Selection favors: (1) employers within the last ~5 years, (2) long
        tenures, and (3) names/summaries that plausibly relate to the current
        target company. Returns employer names in research priority order.
        """
        raw_lines = state.candidate.raw_resume_text.splitlines()

        def _date_for(company: str) -> str:
            """Find a date line for a company in the raw resume text."""
            for line in raw_lines:
                if company.lower() in line.lower() and re.search(
                    r"(?:19|20)\d{2}[-/.]\d{1,2}\s+(?:to|至)", line
                ):
                    return line.strip()
            return ""

        employers: list[dict[str, Any]] = []
        confirmed_context = "\n".join(state.confirmed_resume_facts()).lower()
        reviewed = bool(state.resume_review.claims)
        experience = state.candidate.experience or []
        if experience:
            for e in experience:
                company = str(e.get("company", "")).strip()
                if not company:
                    continue
                start = str(e.get("duration") or e.get("date_range") or "").strip()
                if not re.search(r"(?:19|20)\d{2}", start):
                    start = _date_for(company) or start
                employers.append(
                    {
                        "company": company,
                        "role": str(e.get("role", "")).strip(),
                        "start": start,
                        "summary": str(e.get("summary", "")).strip(),
                    }
                )
        if reviewed:
            employers = [
                entry for entry in employers if entry["company"].lower() in confirmed_context
            ]
        if not employers and not reviewed:
            # Fall back to parsing raw text lines "YYYY-MM to COMPANY".
            for line in raw_lines:
                match = re.search(
                    r"(?:19|20)\d{2}[-/.]\d{1,2}\s+(?:to|至)\s+([A-Za-z一-鿿][A-Za-z一-鿿0-9 &,.]+?)"
                    r"(?:\s+(?:19|20)\d{2}[-/.]\d{1,2})?$",
                    line.strip(),
                )
                if match:
                    company = match.group(1).strip()
                    if company and len(company) >= 2:
                        employers.append({"company": company, "start": line.strip()})
        # Filter: within ~5 years OR long tenure OR name overlap with target company.
        current = state.company.name or ""
        current_tokens = set(re.findall(r"[A-Za-z0-9]+", current.lower()))
        current_terms = set(re.findall(r"[一-鿿]{2,}", current))
        candidates: list[dict[str, Any]] = []
        for entry in employers:
            company = entry["company"]
            start = entry.get("start") or ""
            years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", start)]
            if not years:
                # A dateless entry may still be a recent employer (LLM wrote
                # "3 years" instead of a date range). Keep it, lowest priority.
                name_tokens = set(re.findall(r"[A-Za-z0-9]+", company.lower()))
                name_terms = set(re.findall(r"[一-鿿]{2,}", company))
                entry["_recent"] = False
                entry["_tenure"] = 0
                entry["_related"] = bool(
                    (name_tokens & current_tokens) or (name_terms & current_terms)
                )
                candidates.append(entry)
                continue
            recent = bool(max(years) >= (datetime.now(timezone.utc).year - 5))  # within ~5 yrs
            tenure_span = max(years) - min(years)
            long_tenure = tenure_span >= 3
            name_tokens = set(re.findall(r"[A-Za-z0-9]+", company.lower()))
            name_terms = set(re.findall(r"[一-鿿]{2,}", company))
            related = bool((name_tokens & current_tokens) or (name_terms & current_terms))
            if recent or long_tenure or related:
                entry["_recent"] = recent
                entry["_tenure"] = tenure_span
                entry["_related"] = related
                candidates.append(entry)

        # Priority: recent and related > long tenure > recency alone.
        def priority(entry: dict[str, Any]) -> tuple[int, int]:
            p = 0
            if entry["_recent"]:
                p += 4
            if entry["_related"]:
                p += 2
            if entry["_tenure"] >= 3:
                p += 1
            return (p, entry["_tenure"])

        candidates.sort(key=priority, reverse=True)
        return [entry["company"] for entry in candidates[:limit]]

    async def _research_employers_inline(self, state: InterviewState) -> str:
        """Research recent past employers and return a prompt block (no persist).

        Used inside a running workflow where the session lock is already held.
        Returns an empty string when research is not allowed or yields nothing.
        """
        # Past-employer research touches sensitive candidate data, so it requires
        # explicit public-research authorization (no implicit consent in
        # interactive mode).
        if not state.autopilot.authorized_public_research:
            return ""
        employers = self._recent_employers(state, limit=2)
        collected: list[dict[str, Any]] = []
        for company in employers:
            collected.extend(await self._search_employer(company))
        existing = filter_employer_business_results(state.past_employer_sources, entity="")
        merged = merge_search_results(existing, collected, limit=8)
        if not merged:
            state.past_employer_sources = []
            state.past_employer_block = ""
            return ""
        state.past_employer_sources = merged
        return format_employer_business_context(merged)

    async def _search_employer(self, company: str) -> list[dict[str, Any]]:
        """Find business-background sources for one past employer.

        This path deliberately never queries current jobs or recruiting pages:
        those describe the employer's present vacancies, not the candidate's
        historical responsibilities.
        """
        provider = self.search_provider
        if provider is None:
            return []
        queries = [
            f'"{company}" 官方网站 公司简介 主要业务 产品 服务',
            f'"{company}" official website about products services business',
        ]
        results: list[dict[str, Any]] = []
        for query in queries:
            try:
                batch = await provider.search(query, limit=5, search_depth="basic")
            except Exception as exc:  # noqa: BLE001 - provider may be disabled
                logger.warning("Past-employer search failed (%s)", type(exc).__name__)
                continue
            batch_dicts: list[dict[str, Any]] = []
            for item in batch:
                if isinstance(item, dict):
                    batch_dicts.append(item)
                else:
                    batch_dicts.append(item.model_dump(mode="json"))
            filtered = filter_entity_results(batch_dicts, entity=company)
            if filtered:
                results = merge_search_results(results, filtered, limit=8)
                if any(
                    item.get("is_official") or item.get("source_quality") == "official"
                    for item in filtered
                ):
                    break
        return filter_employer_business_results(results, entity=company)

    async def research_recent_employers(self, session_id: str) -> InterviewState:
        """Search the candidate's recent/important past employers and persist sources.

        Runs inside the session lock; stores merged sources, sets the research
        status, rebuilds fact cards/entity resolutions, and persists. Search is
        gated by the same public-research consent as company research.
        """
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            state = runtime.state
            if not state.autopilot.authorized_public_research:
                state.past_employer_research_status = "consent_required"
                await self._persist(session_id, state)
                return state
            employers = self._recent_employers(state, limit=2)
            collected: list[dict[str, Any]] = []
            state.past_employer_sources = filter_employer_business_results(
                state.past_employer_sources, entity=""
            )
            for company in employers:
                if any(
                    company.lower() in str(source.get("title", "")).lower()
                    or company.lower() in str(source.get("snippet", "")).lower()
                    for source in state.past_employer_sources
                ):
                    continue  # already researched by the inline workflow path
                sources = await self._search_employer(company)
                collected.extend(sources)
            state.past_employer_sources = merge_search_results(
                state.past_employer_sources, collected, limit=8
            )
            state.past_employer_research_status = (
                "completed" if state.past_employer_sources else "no_results"
            )
            state.past_employer_block = format_employer_business_context(
                state.past_employer_sources
            )
            self._sync_intelligence(state)
            await self._persist(session_id, state)
        self._record_debug(
            "past_employer_research",
            session_id,
            detail=f"employer_count={len(employers)}; sources={len(state.past_employer_sources)}",
        )
        return runtime.state

    async def analyze_interviewer(
        self,
        session_id: str,
        name: str,
        position: str = "",
        company: str = "",
        public_info: str = "",
    ) -> Message:
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            runtime.state.interviewer = InterviewerProfile(
                name=name,
                position=position,
                company=company,
                public_expressions=[{"text": public_info}] if public_info else [],
            )
            message = await runtime.run("interviewer_agent")
            self._sync_intelligence(runtime.state)
            await self._persist(session_id, runtime.state)
            return message

    async def run_candidate_prep(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        interviewer_name: str = "",
        interviewer_position: str = "",
        interviewer_public_info: str = "",
        authorized_public_research: bool = False,
        prepare_resume_transition: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        if prepare_resume_transition:
            await self._prepare_resume_transition(session_id, runtime, resume_text)
        job_input, job_sources, job_research_status = await self._enrich_title_only_job_input(
            job_description, company_name, authorized=authorized_public_research
        )
        steps: list[tuple[str, str]] = [
            ("candidate_agent", resume_text),
            ("job_agent", job_input),
            ("company_agent", company_context),
        ]
        interviewer = None
        if interviewer_name:
            interviewer = InterviewerProfile(
                name=interviewer_name,
                position=interviewer_position,
                company=company_name,
                public_expressions=(
                    [{"text": interviewer_public_info}] if interviewer_public_info else []
                ),
            )
            steps.append(("interviewer_agent", ""))
        steps.extend(
            [
                ("interview_strategy_agent", "Generate personalized interview strategy"),
                ("mock_interview_agent", "Generate personalized mock questions"),
            ]
        )
        try:
            state = await self._execute_workflow(
                session_id,
                runtime,
                "candidate_prep",
                steps,
                company_name=company_name,
                company_context=company_context,
                interviewer=interviewer,
                parallel_prefix=4 if interviewer else 3,
                authorized_public_research=authorized_public_research,
            )
        except Exception:
            # JobAgent may already have persisted the internal source-enriched
            # prompt before a later strategy/question step fails. Restore the
            # user's original title and provenance on the failure path too.
            await self._attach_job_research(
                session_id,
                runtime.state,
                original_input=job_description,
                sources=job_sources,
                status=job_research_status,
            )
            raise
        await self._attach_job_research(
            session_id,
            state,
            original_input=job_description,
            sources=job_sources,
            status=job_research_status,
        )
        return state

    async def run_enterprise_design(
        self,
        session_id: str,
        *,
        resume_text: str,
        job_description: str,
        company_name: str,
        company_context: str = "",
        authorized_public_research: bool = False,
        prepare_resume_transition: bool = True,
    ) -> InterviewState:
        runtime = await self._get_runtime(session_id)
        if prepare_resume_transition:
            await self._prepare_resume_transition(session_id, runtime, resume_text)
        job_input, job_sources, job_research_status = await self._enrich_title_only_job_input(
            job_description, company_name, authorized=authorized_public_research
        )
        steps = [
            ("candidate_agent", resume_text),
            ("job_agent", job_input),
            ("company_agent", company_context),
            ("interview_design_agent", "Design an evidence-based interview blueprint"),
        ]
        try:
            state = await self._execute_workflow(
                session_id,
                runtime,
                "enterprise_design",
                steps,
                company_name=company_name,
                company_context=company_context,
                parallel_prefix=3,
                authorized_public_research=authorized_public_research,
            )
        except Exception:
            await self._attach_job_research(
                session_id,
                runtime.state,
                original_input=job_description,
                sources=job_sources,
                status=job_research_status,
            )
            raise
        await self._attach_job_research(
            session_id,
            state,
            original_input=job_description,
            sources=job_sources,
            status=job_research_status,
        )
        return state

    async def _enrich_title_only_job_input(
        self, job_description: str, company_name: str, *, authorized: bool
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Resolve a bare title into source-bound public JD context before planning."""
        initial_review = review_job_description(job_description, [])
        if not initial_review.is_title_only:
            return job_description, [], "not_needed"
        if not authorized:
            return job_description, [], "consent_required"
        if self.search_provider is None:
            return job_description, [], "not_configured"
        title = job_description.strip()
        company = company_name.strip()
        query = f'"{title}"'
        if company:
            query += f' "{company}"'
        query += " 岗位职责 任职要求 招聘 JD"
        try:
            results = await self.search_provider.search(query, limit=6, search_depth="advanced")
        except Exception as exc:  # noqa: BLE001 - provider boundary
            logger.warning("Public JD research failed (%s)", type(exc).__name__)
            return job_description, [], "failed"
        title_tokens = {
            token.casefold() for token in re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", title)
        }
        relevant = []
        for result in results:
            payload = result.model_dump(mode="json")
            haystack = f"{result.title} {result.snippet}".casefold()
            title_match = any(token in haystack for token in title_tokens)
            if title_match:
                relevant.append(payload)
        sources = relevant[:5]
        if not sources:
            return job_description, [], "no_reliable_sources"
        context = format_search_results(sources)
        enriched = (
            f"用户输入的职位名称：{title}\n"
            f"目标公司：{company or '未指定'}\n\n"
            "以下是公开招聘结果的候选 JD 摘要，未经用户确认；只能用于生成待核验的岗位问题，"
            "不得表述为目标公司的明确要求：\n"
            f"{context}"
        )
        return enriched, sources, "completed"

    async def _attach_job_research(
        self,
        session_id: str,
        state: InterviewState,
        *,
        original_input: str,
        sources: list[dict[str, Any]],
        status: str,
    ) -> None:
        if status == "not_needed":
            return
        async with self._lock_for(session_id):
            state.job.raw_description = original_input
            state.job_review.is_title_only = True
            state.job_review.completeness_score = 0.55 if sources else 0.2
            state.job_review.public_sources = sources
            state.job_review.public_research_status = status
            state.job_review.researched_title = original_input.strip()
            state.job_review.requirements = [
                item.model_copy(update={"origin": RequirementOrigin.INFERRED})
                for item in state.job_review.requirements
            ]
            warning = (
                "已根据公开招聘来源补全候选 JD；职责与要求仍是待确认推测。"
                if sources
                else "需要允许公开检索后才能根据职位名称补全 JD。"
                if status == "consent_required"
                else "未找到可靠公开 JD；当前问题只能按职位名称生成通用准备方向。"
            )
            state.job_review.warnings = list(dict.fromkeys([warning, *state.job_review.warnings]))
            await self._persist(session_id, state)
        self._record_debug(
            "job_jd_research_completed",
            session_id,
            detail=f"status={status}; sources={len(sources)}",
        )
