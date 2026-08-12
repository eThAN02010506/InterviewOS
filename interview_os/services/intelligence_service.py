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
from interview_os.tools.web_search import filter_employer_business_results


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
    sources = [
        *state.company.public_sources,
        *state.interviewer.public_expressions,
        *state.past_employer_sources,
    ]
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
    indexed: dict[tuple[str, str], FactCard] = {}
    batches: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("company", state.company.name, state.company.public_sources),
        ("interviewer", state.interviewer.name, state.interviewer.public_expressions),
    ]
    if state.past_employer_sources:
        employer_sources = filter_employer_business_results(
            state.past_employer_sources, entity=""
        )
        if employer_sources:
            batches.append(("past_employer", "过往雇主业务背景", employer_sources))
    for category, subject, sources in batches:
        for source in sources:
            url = str(source.get("url", "")).strip()
            claim = _fact_claim(source, subject)
            normalized_claim = re.sub(r"[^\w\u4e00-\u9fff]", "", claim.lower())[:120]
            key = (category, normalized_claim)
            if not claim:
                continue
            quality = _source_quality(source)
            fetched_at = _source_fetched_at(source)
            filter_reason = str(source.get("filter_reason", "")).strip()
            verified = quality in {"official", "high"}
            existing = indexed.get(key)
            if existing:
                existing.source_count += 1
                if url and url not in existing.source_urls:
                    existing.source_urls.append(url)
                existing.source_quality = _merge_source_quality(
                    existing.source_quality, quality
                )
                if fetched_at and (
                    existing.source_fetched_at is None
                    or fetched_at > existing.source_fetched_at
                ):
                    existing.source_fetched_at = fetched_at
                if filter_reason and filter_reason not in existing.source_filter_reason:
                    existing.source_filter_reason = _append_reason(
                        existing.source_filter_reason, filter_reason
                    )
                existing.cache_hit = existing.cache_hit or bool(source.get("cache_hit"))
                if verified:
                    existing.status = FactStatus.VERIFIED
                existing.confidence = min(0.95, existing.confidence + 0.08)
                existing.note = (
                    f"{existing.source_count} 个公开来源交叉支持，"
                    f"最高来源质量：{_source_quality_label(existing.source_quality)}"
                )
                continue
            card = FactCard(
                category=category if category == "past_employer" else _category_for(category, claim),
                subject=subject,
                claim=claim,
                status=FactStatus.VERIFIED if verified else FactStatus.INFERRED,
                confidence=0.9 if verified else 0.65,
                source_urls=[url] if url else [],
                source_quality=quality,
                source_count=1,
                source_fetched_at=fetched_at,
                source_filter_reason=filter_reason,
                cache_hit=bool(source.get("cache_hit")),
                note=(
                    (
                        "仅描述过往雇主业务，不代表候选人职责或能力；建议打开原文复核"
                        if category == "past_employer"
                        else "来自公开来源摘要，建议打开原文复核"
                    )
                    if not verified
                    else (
                        f"{_source_quality_label(quality)}来源；仅作公司业务背景，"
                        "不作为候选人经历证据"
                        if category == "past_employer"
                        else f"{_source_quality_label(quality)}来源"
                    )
                ),
            )
            indexed[key] = card
            cards.append(card)
    conflict = _position_conflict(state)
    if conflict:
        cards.insert(0, conflict)
    state.fact_cards = cards[:30]


def resolve_entity(
    state: InterviewState, resolution_id, *, accept: bool, proposed_name: str = ""
) -> EntityResolution:
    resolution = next((item for item in state.entity_resolutions if item.id == resolution_id), None)
    if resolution is None:
        raise LookupError("Entity resolution not found")
    clean_name = proposed_name.strip()
    if accept and clean_name:
        resolution.proposed_name = clean_name
    resolution.status = ResolutionStatus.ACCEPTED if accept else ResolutionStatus.REJECTED
    resolution.resolved_at = datetime.now(timezone.utc)
    if accept and resolution.entity_type == "company":
        if state.company.name == resolution.input_name:
            state.company.name = resolution.proposed_name
        if state.interviewer.company == resolution.input_name:
            state.interviewer.company = resolution.proposed_name
    return resolution


def decide_fact_card(
    state: InterviewState, card_id, *, action: str, note: str = ""
) -> FactCard:
    card = next((item for item in state.fact_cards if item.id == card_id), None)
    if card is None:
        raise LookupError("Fact card not found")
    clean_note = note.strip()
    if action == "accept":
        card.status = FactStatus.ACCEPTED
        card.confidence = max(card.confidence, 0.85)
        card.note = clean_note or "用户确认可作为上下文使用"
        card.resolved_at = datetime.now(timezone.utc)
    elif action == "reject":
        card.status = FactStatus.REJECTED
        card.confidence = min(card.confidence, 0.2)
        card.note = clean_note or "用户已排除，不作为后续上下文依据"
        card.resolved_at = datetime.now(timezone.utc)
    elif action == "reset":
        card.status = FactStatus.INFERRED
        card.confidence = 0.65
        card.note = clean_note or "已恢复为待复核推测"
        card.resolved_at = None
    else:
        raise ValueError("Unsupported fact card decision")
    return card


def _fact_claim(source: dict[str, Any], subject: str) -> str:
    text = _clean_public_text(str(source.get("snippet") or source.get("text") or ""))
    sentences = [
        item.strip(" -—:：")
        for item in re.split(r"(?<=[。！？.!?])\s+|[\r\n]+", text)
        if 18 <= len(item.strip()) <= 220
    ]
    boilerplate = ("首页", "登录", "注册", "搜索", "新闻 专栏", "Image ", "全部删除")
    image_caption = re.compile(
        r"^(?:an?\s+)?(?:image|photo|headshot|portrait|woman|man|person)\b|"
        r"\b(?:wearing|smiling|arms crossed|plain (?:light |dark )?background)\b",
        re.IGNORECASE,
    )
    candidates = [
        item
        for item in sentences
        if sum(token in item for token in boilerplate) < 2
        and not image_caption.search(item)
    ]
    if candidates:
        candidates.sort(
            key=lambda item: (
                subject.lower() in item.lower(),
                bool(
                    re.search(
                        r"创始|技术|负责|研发|产品|观点|表示|提出|engineer|founder",
                        item,
                        re.IGNORECASE,
                    )
                ),
                -abs(len(item) - 100),
            ),
            reverse=True,
        )
        return candidates[0][:220]
    # Page titles remain available in the source list, but they are not a
    # factual claim by themselves. Do not promote a title or image caption into
    # a verified fact card.
    return ""


def _clean_public_text(value: str) -> str:
    value = re.sub(r"Image\s*\d*:?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[#*_`]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _source_quality(source: dict[str, Any]) -> str:
    if bool(source.get("is_official")):
        return "official"
    quality = str(source.get("source_quality") or "unrated").strip().lower()
    return quality if quality in {"official", "high", "secondary", "unrated"} else "unrated"


def _merge_source_quality(current: str, incoming: str) -> str:
    order = {"official": 4, "high": 3, "secondary": 2, "unrated": 1}
    normalized_current = current if current in order else "unrated"
    normalized_incoming = incoming if incoming in order else "unrated"
    if normalized_current == normalized_incoming:
        return normalized_current
    if "official" in {normalized_current, normalized_incoming}:
        return "official"
    return max((normalized_current, normalized_incoming), key=lambda item: order[item])


def _source_quality_label(quality: str) -> str:
    return {
        "official": "官方",
        "high": "高可信",
        "secondary": "二级",
        "unrated": "未评级",
    }.get(quality, "未评级")


def _source_fetched_at(source: dict[str, Any]) -> datetime | None:
    raw_value = source.get("fetched_at")
    if isinstance(raw_value, datetime):
        return raw_value if raw_value.tzinfo else raw_value.replace(tzinfo=timezone.utc)
    if not raw_value:
        return None
    try:
        value = str(raw_value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _append_reason(current: str, reason: str) -> str:
    if not current:
        return reason
    return f"{current}; {reason}"[:500]


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
    fetched_times = [
        fetched
        for item in state.interviewer.public_expressions
        if (fetched := _source_fetched_at(item)) is not None
    ]
    return FactCard(
        category="interviewer",
        subject=state.interviewer.name,
        claim=f"用户输入职位为“{supplied}”，公开资料主要出现“{'、'.join(dict.fromkeys(roles))}”。",
        status=FactStatus.CONFLICT,
        confidence=0.85,
        source_urls=list(dict.fromkeys(urls))[:5],
        source_quality="secondary",
        source_count=len(state.interviewer.public_expressions),
        source_fetched_at=max(fetched_times) if fetched_times else None,
        source_filter_reason="conflict: user supplied role differs from public-source role terms",
        cache_hit=any(bool(item.get("cache_hit")) for item in state.interviewer.public_expressions),
        note="职位信息存在冲突，不能自动覆盖，请人工确认。",
    )
