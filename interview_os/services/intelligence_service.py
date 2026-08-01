"""Deterministic review models derived from user input and search provenance."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from interview_os.core.state import (
    EntityResolution,
    FactCard,
    FactStatus,
    InterviewState,
    JobDescriptionReview,
    JobRequirement,
    RequirementOrigin,
    ResolutionStatus,
)


def review_job_description(raw_text: str, inferred: list[str]) -> JobDescriptionReview:
    text = raw_text.strip()
    title_only = (
        bool(text)
        and len(text) <= 120
        and "\n" not in text
        and not re.search(
            r"职责|要求|任职|岗位|responsibilit|requirement|qualification|you will|must have",
            text,
            flags=re.IGNORECASE,
        )
    )
    section_markers = {
        "岗位职责": r"职责|工作内容|responsibilit|you will|what you'll do",
        "任职要求": r"任职|要求|资格|requirement|qualification|must have",
        "团队背景": r"团队|汇报|协作|部门|team|reporting|department",
    }
    present = {
        name for name, pattern in section_markers.items() if re.search(pattern, text, re.IGNORECASE)
    }
    missing = (
        list(section_markers)
        if title_only
        else [name for name in section_markers if name not in present]
    )
    lines = [
        re.sub(r"^[\s•·*\-—\d.、)）]+", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    explicit = [] if title_only else [line for line in lines if len(line) >= 6][:30]
    requirements = [
        JobRequirement(text=line, origin=RequirementOrigin.EXPLICIT) for line in explicit
    ]
    requirements.extend(
        JobRequirement(text=item, origin=RequirementOrigin.INFERRED)
        for item in inferred
        if item and item not in explicit
    )
    completeness = 0.2 if title_only else min(1.0, 0.4 + len(present) * 0.2)
    warnings = []
    if title_only:
        warnings.append("当前输入只有职位名称；生成内容仅是通用准备方向，不代表公司明确要求。")
    elif missing:
        warnings.append(f"建议补充：{'、'.join(missing)}。")
    return JobDescriptionReview(
        is_title_only=title_only,
        completeness_score=completeness,
        missing_sections=missing,
        requirements=requirements,
        warnings=warnings,
    )


def sync_entity_resolutions(state: InterviewState) -> None:
    sources = [*state.company.public_sources, *state.interviewer.public_expressions]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        if source.get("identity_match") != "corroborated_alias":
            continue
        input_name = str(source.get("input_identity", "")).strip()
        proposed = str(source.get("matched_identity", "")).strip()
        if input_name and proposed and input_name != proposed:
            grouped.setdefault((input_name, proposed), []).append(source)
    for (input_name, proposed), evidence in grouped.items():
        existing = next(
            (
                item
                for item in state.entity_resolutions
                if item.input_name == input_name and item.proposed_name == proposed
            ),
            None,
        )
        urls = list(dict.fromkeys(str(item.get("url", "")) for item in evidence if item.get("url")))
        if existing:
            existing.source_urls = urls
            continue
        state.entity_resolutions.append(
            EntityResolution(
                entity_type="company",
                input_name=input_name,
                proposed_name=proposed,
                confidence=min(0.98, 0.72 + 0.04 * len(urls)),
                reason="相近名称由同一面试官身份及多个公开来源交叉验证",
                source_urls=urls,
            )
        )


def build_fact_cards(state: InterviewState) -> None:
    cards: list[FactCard] = []
    seen: set[tuple[str, str]] = set()
    batches = (
        ("company", state.company.name, state.company.public_sources),
        ("interviewer", state.interviewer.name, state.interviewer.public_expressions),
    )
    for category, subject, sources in batches:
        for source in sources:
            url = str(source.get("url", "")).strip()
            claim = _fact_claim(source)
            key = (category, url or claim)
            if not claim or key in seen:
                continue
            seen.add(key)
            quality = str(source.get("source_quality", "unrated"))
            verified = bool(source.get("is_official")) or quality in {"official", "high"}
            cards.append(
                FactCard(
                    category=_category_for(category, claim),
                    subject=subject,
                    claim=claim,
                    status=FactStatus.VERIFIED if verified else FactStatus.INFERRED,
                    confidence=0.9 if verified else 0.65,
                    source_urls=[url] if url else [],
                    note="来自公开来源摘要，建议打开原文复核" if not verified else "高可信来源",
                )
            )
    conflict = _position_conflict(state)
    if conflict:
        cards.insert(0, conflict)
    state.fact_cards = cards[:30]


def resolve_entity(state: InterviewState, resolution_id, *, accept: bool) -> EntityResolution:
    resolution = next((item for item in state.entity_resolutions if item.id == resolution_id), None)
    if resolution is None:
        raise LookupError("Entity resolution not found")
    resolution.status = ResolutionStatus.ACCEPTED if accept else ResolutionStatus.REJECTED
    resolution.resolved_at = datetime.now(timezone.utc)
    if accept and resolution.entity_type == "company":
        if state.company.name == resolution.input_name:
            state.company.name = resolution.proposed_name
        if state.interviewer.company == resolution.input_name:
            state.interviewer.company = resolution.proposed_name
    return resolution


def _fact_claim(source: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(source.get("snippet") or source.get("text") or "")).strip()
    title = re.sub(r"\s+", " ", str(source.get("title", ""))).strip()
    if text:
        sentence = re.split(r"(?<=[。！？.!?])\s*", text, maxsplit=1)[0]
        return sentence[:280]
    return title[:280]


def _category_for(default: str, claim: str) -> str:
    if re.search(r"技术|芯片|光谱|AI|人工智能|engineering|technology", claim, re.IGNORECASE):
        return "technology"
    if re.search(r"表示|认为|强调|提出|演讲|采访|said|believes", claim, re.IGNORECASE):
        return "public_view"
    return default


def _position_conflict(state: InterviewState) -> FactCard | None:
    supplied = state.interviewer.position.strip()
    if not supplied:
        return None
    text = " ".join(
        f"{item.get('title', '')} {item.get('snippet', '')}"
        for item in state.interviewer.public_expressions
    )
    if supplied.lower() in text.lower():
        return None
    roles = [
        role
        for role in ("创始人", "首席科学家", "董事长", "教授", "CTO", "CEO")
        if role.lower() in text.lower() and role.lower() != supplied.lower()
    ]
    if not roles:
        return None
    urls = [
        str(item.get("url")) for item in state.interviewer.public_expressions if item.get("url")
    ]
    return FactCard(
        category="interviewer",
        subject=state.interviewer.name,
        claim=f"用户输入职位为“{supplied}”，公开资料主要出现“{'、'.join(dict.fromkeys(roles))}”。",
        status=FactStatus.CONFLICT,
        confidence=0.85,
        source_urls=list(dict.fromkeys(urls))[:5],
        note="职位信息存在冲突，不能自动覆盖，请人工确认。",
    )
