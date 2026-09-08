"""Deterministic, evidence-led follow-up planning for mock interviews.

The language model may enrich the wording of a question, but the decision to
probe must remain inspectable and fast.  This planner selects the next branch
from the persisted question graph by looking at the latest answer's coverage
contract.  It never invents a candidate fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from interview_os.core.question_understanding import deterministic_question_understanding
from interview_os.core.state import InterviewQuestion, MockAnswerRecord, QuestionProbeNode

MAX_AUTOMATIC_FOLLOW_UPS = 4


@dataclass(frozen=True)
class FollowUpDecision:
    question: str
    stage: str
    rationale: str


_UNABLE_PATTERNS = re.compile(
    r"^(?:呃[，, ]*)?(?:我)?(?:不知道|不清楚|不会|没做过|没有做过|没有经验|"
    r"想不起来|不确定)(?:了|这个|这方面|。|\.|[，, ]*)?$|"
    r"^(?:i )?(?:do not|don't) know|^(?:i )?(?:have not|haven't) done",
    re.IGNORECASE,
)


def plan_follow_up(
    question: InterviewQuestion,
    latest_response: MockAnswerRecord,
    *,
    offered_questions: list[str] | None = None,
) -> FollowUpDecision | None:
    """Select one unused, answer-grounded follow-up branch.

    A maximum keeps a single question from consuming the whole interview.  The
    user can still skip any offered probe through the existing next action.
    """

    offered = [item.strip() for item in (offered_questions or []) if item.strip()]
    if len(offered) >= MAX_AUTOMATIC_FOLLOW_UPS:
        return None

    answer = latest_response.answer.strip()
    if _UNABLE_PATTERNS.search(answer):
        recovery = _recovery_question(question)
        if recovery not in offered:
            return FollowUpDecision(
                question=recovery,
                stage="recovery",
                rationale="候选人表示暂时无法回答；先区分缺少经历与不会组织表达。",
            )

    understanding = question.understanding or deterministic_question_understanding(
        question.question,
        competency=question.competency,
    )
    preferred_stage, rationale = _stage_from_coverage(latest_response)
    nodes = {node.stage: node for node in understanding.probe_tree if node.question.strip()}

    # Start at the answer-grounded gap, then deepen in a stable order.  Already
    # offered wording is never repeated, including after session restoration.
    order = _ordered_stages(preferred_stage)
    for stage in order:
        node = nodes.get(stage)
        if node is not None and node.question.strip() not in offered:
            return _decision_from_node(node, rationale if stage == preferred_stage else "继续验证回答抗追问性。")

    # Legacy/generated questions may not yet have a deep tree.  Preserve their
    # authored follow-ups as a bounded fallback rather than dropping the probe.
    if not nodes:
        for candidate in question.follow_ups:
            text = candidate.strip()
            if text and text not in offered:
                return FollowUpDecision(
                    question=text,
                    stage=preferred_stage,
                    rationale=rationale,
                )
    return None


def _stage_from_coverage(response: MockAnswerRecord) -> tuple[str, str]:
    coverage = response.evaluation.spoken_analysis.question_coverage
    incomplete = {
        item.requirement
        for item in coverage
        if item.status in {"missing", "partial"}
    }
    if "直接回应题目核心" in incomplete:
        return "foundation", "回答尚未直接形成针对题目的核心结论。"
    if incomplete & {
        "提供一个真实案例",
        "明确具体公司/业务场景",
        "明确个人职责与关键决策",
        "说明指标、口径与决策关系",
        "说明执行动作",
        "给出结果与验证方式",
    }:
        return "evidence", "当前结论缺少可归属于候选人的事实或验证证据。"
    if incomplete & {
        "说明备选方案、权衡标准与最终选择",
        "说明方法或制定过程",
    }:
        return "tradeoff", "回答给出了方向，但关键取舍和决策门槛仍不清楚。"
    if incomplete or not coverage:
        return "foundation", "当前问题的核心要求仍待澄清，不能视为已经覆盖。"
    return "pressure", "核心要求已覆盖，继续测试边界、反事实和迁移能力。"


def _ordered_stages(preferred: str) -> list[str]:
    stages = ["foundation", "evidence", "tradeoff", "pressure"]
    try:
        start = stages.index(preferred)
    except ValueError:
        start = 0
    # Never become shallower merely because a deeper probe was already used.
    return stages[start:]


def _decision_from_node(node: QuestionProbeNode, rationale: str) -> FollowUpDecision:
    return FollowUpDecision(
        question=node.question.strip(),
        stage=node.stage,
        rationale=rationale or node.purpose,
    )


def _recovery_question(question: InterviewQuestion) -> str:
    competency = question.competency.strip() or "这类问题"
    return (
        f"你是完全没有{competency}的直接经历，还是有相近经历但暂时不知道怎么组织？"
        "请选一种说明；没有直接经历时不要编造。"
    )
