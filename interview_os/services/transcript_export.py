"""Render a complete interview transcript as plain text."""

from __future__ import annotations

from interview_os.core.evidence import EvidenceSource
from interview_os.core.state import (
    InterviewState,
    TranscriptSpeaker,
)


def _speaker_label(speaker: TranscriptSpeaker) -> str:
    if speaker == TranscriptSpeaker.CANDIDATE:
        return "候选人"
    if speaker == TranscriptSpeaker.INTERVIEWER:
        return "面试官"
    return "待确认"


def _fmt(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "—"


def render_transcript(state: InterviewState) -> str:
    """Render the full interview record for a session as UTF-8 text."""
    lines: list[str] = []
    candidate = state.candidate.name or "未命名候选人"
    job = state.job.title or "未指定岗位"
    company = state.company.name or ""
    lines.append("=" * 60)
    lines.append("InterviewOS 面试记录")
    lines.append("=" * 60)
    lines.append(f"候选人：{candidate}")
    lines.append(f"目标岗位：{job}")
    if company:
        lines.append(f"公司：{company}")
    live = state.live_interview
    if live.started_at:
        lines.append(f"面试开始：{_fmt(live.started_at)}")
    if live.completed_at:
        lines.append(f"面试结束：{_fmt(live.completed_at)}")
    lines.append("")

    if live.rolling_summary:
        lines.append("【长面试摘要】")
        lines.append(live.rolling_summary)
        lines.append("")

    if live.segments:
        lines.append("【对话转写】")
        for segment in live.segments:
            label = _speaker_label(segment.speaker)
            time_str = _fmt(segment.started_at) if segment.started_at else ""
            source = segment.source or "manual"
            stable = "" if (segment.stable and segment.confirmed) else "（草稿）"
            lines.append(f"[{time_str}] {label}（{source}）{stable}")
            lines.append(f"    {segment.text}")
        lines.append("")

    if live.suggestions:
        lines.append("【AI 问题建议】")
        status_labels = {
            "pending": "待处理",
            "adopted": "已采用",
            "edited": "编辑后采用",
            "skipped": "已跳过",
        }
        for suggestion in live.suggestions:
            question = suggestion.final_question or suggestion.suggested_question
            status = status_labels.get(suggestion.status, suggestion.status)
            lines.append(
                f"- [{status}] {suggestion.question_type} · {suggestion.competency}"
            )
            lines.append(f"    {question}")
            if suggestion.rationale:
                lines.append(f"    理由：{suggestion.rationale}")
            if suggestion.evidence_gap:
                lines.append(f"    待补证据：{suggestion.evidence_gap}")
            if suggestion.expected_signals:
                lines.append(f"    预期信号：{'、'.join(suggestion.expected_signals)}")
        lines.append("")

    records = state.live_interview_records
    if records:
        lines.append("【已确认证据】")
        for record in records:
            score = ""
            if record.scoring_status == "scored":
                score = f"（均分 {record.evaluation.overall_score():.2f}）"
            elif record.scoring_status == "failed":
                score = "（评分失败）"
            lines.append(f"- {record.competency} {score}")
            lines.append(f"    问题：{record.question}")
            lines.append(f"    回答：{record.answer}")
        lines.append("")

    evidence = [
        item for item in state.evidence if item.source == EvidenceSource.LIVE_INTERVIEW
    ]
    if evidence:
        lines.append("【能力信号汇总】")
        for item in evidence:
            lines.append(
                f"- {item.competency} · 置信度 {item.confidence:.2f} · {item.signal}"
            )
        lines.append("")

    lines.append("=" * 60)
    lines.append("本记录为 InterviewOS 自动整理，请面试官审阅后使用。")
    return "\n".join(lines)
