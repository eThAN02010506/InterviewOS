"""Run a repeatable real-HTTP Live Copilot action-card workflow.

This script verifies the interviewer-side live path against a running local
InterviewOS service. It intentionally uses synthetic interview content and the
configured model provider; it is a release smoke test, not a unit-test
replacement.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx


def request_json(
    client: httpx.Client,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.request(method, path, json=payload)
    response.raise_for_status()
    return response.json()


def live_state(data: dict[str, Any]) -> dict[str, Any]:
    return data["state"]["live_interview"]


def action_type(data: dict[str, Any]) -> str:
    return live_state(data)["action_card"]["action_type"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def append_segment(
    client: httpx.Client,
    session_id: str,
    *,
    speaker: str,
    text: str,
) -> dict[str, Any]:
    return request_json(
        client,
        "POST",
        f"/api/live-interviews/{session_id}/segments",
        {"speaker": speaker, "text": text},
    )


def last_segment(data: dict[str, Any]) -> dict[str, Any]:
    return live_state(data)["segments"][-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    started_at = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    def checkpoint(name: str, data: dict[str, Any]) -> None:
        card = live_state(data)["action_card"]
        checkpoints.append(
            {
                "name": name,
                "action": card["action_type"],
                "priority": card["priority"],
                "primary_cta": card["primary_cta"],
                "evidence_status": card["evidence_status"],
            }
        )

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        session = request_json(
            client,
            "POST",
            "/api/interviews/sessions",
            {
                "candidate_name": "Live Action E2E 候选人",
                "job_title": "高级平台工程师",
                "company_name": "E2E Test Company",
            },
        )
        session_id = session["id"]

        phase_started = time.perf_counter()
        started = request_json(
            client,
            "POST",
            f"/api/live-interviews/{session_id}/start",
            {"consent_confirmed": True},
        )
        checkpoint("started", started)
        require(action_type(started) == "plan_gap_question", "start should ask for evidence")

        first_question = append_segment(
            client,
            session_id,
            speaker="interviewer",
            text="请讲一次你主导的平台稳定性治理项目。",
        )
        question_id = last_segment(first_question)["id"]
        first_answer = append_segment(
            client,
            session_id,
            speaker="candidate",
            text="第一阶段我先梳理订单服务的错误预算、报警噪声和慢查询，把风险按影响面排序。",
        )
        first_answer_id = last_segment(first_answer)["id"]
        second_answer = append_segment(
            client,
            session_id,
            speaker="candidate",
            text=(
                "第二阶段我补了限流、灰度、回滚预案和事故复盘机制，"
                "最终高峰错误率下降 60%，以上。"
            ),
        )
        checkpoint("boundary_detected", second_answer)
        require(action_type(second_answer) == "merge_boundary", "two answer chunks should merge")
        boundary = live_state(second_answer)["answer_boundary_suggestions"][0]
        require(
            boundary["answer_segment_ids"] == [first_answer_id, last_segment(second_answer)["id"]],
            "boundary suggestion should target the two candidate answer chunks",
        )

        merged = request_json(
            client,
            "POST",
            f"/api/live-interviews/{session_id}/evidence/merge",
            {
                "segment_ids": boundary["answer_segment_ids"],
                "question_segment_id": question_id,
                "competency": "Reliability",
            },
        )
        checkpoint("merged_evidence", merged)
        # Merging evidence now auto-plans the next question, so the card moves
        # straight to deciding a pending suggestion (no manual plan step).
        require(action_type(merged) == "decide_question", "merge should auto-plan a suggestion")
        require(len(merged["state"]["live_interview_records"]) == 1, "one live record expected")
        suggestion = live_state(merged)["suggestions"][-1]
        require(suggestion["status"] == "pending", "auto-planned suggestion should be pending")

        adopted = request_json(
            client,
            "PATCH",
            f"/api/live-interviews/{session_id}/suggestions/{suggestion['id']}",
            {"status": "adopted"},
        )
        checkpoint("question_adopted", adopted)
        require(
            live_state(adopted)["segments"][-1]["speaker"] == "interviewer",
            "adopting a suggestion should append an interviewer question",
        )
        timings["live_action_card_seconds"] = round(time.perf_counter() - phase_started, 3)

        second_response = append_segment(
            client,
            session_id,
            speaker="candidate",
            text=(
                "这个追问里，我会补充容量压测方法：先回放线上流量，再用阶梯压测确认瓶颈，"
                "最后把连接池、缓存命中率和降级阈值写进发布检查表。"
            ),
        )
        second_record = request_json(
            client,
            "POST",
            f"/api/live-interviews/{session_id}/segments/{last_segment(second_response)['id']}/evidence",
            {"competency": "System Design"},
        )
        checkpoint("second_evidence", second_record)

        third_question = append_segment(
            client,
            session_id,
            speaker="interviewer",
            text="再讲一个跨团队推进或冲突处理的案例。",
        )
        third_answer = append_segment(
            client,
            session_id,
            speaker="candidate",
            text=(
                "我推动 SRE、交易和数据团队统一错误预算口径，每周同步风险和 Owner，"
                "对分歧用用户影响、恢复时间和实现成本排序，最后把发布阻塞从两天降到半天。"
            ),
        )
        third_record = request_json(
            client,
            "POST",
            f"/api/live-interviews/{session_id}/segments/{last_segment(third_answer)['id']}/evidence",
            {
                "question_segment_id": last_segment(third_question)["id"],
                "competency": "Collaboration",
            },
        )
        checkpoint("third_evidence", third_record)

        completed = request_json(
            client,
            "POST",
            f"/api/live-interviews/{session_id}/status",
            {"status": "completed"},
        )
        checkpoint("completed", completed)
        require(action_type(completed) == "evaluate", "completed sufficient evidence should evaluate")

        phase_started = time.perf_counter()
        evaluated = request_json(client, "POST", f"/api/evaluations/{session_id}")
        timings["evaluation_seconds"] = round(time.perf_counter() - phase_started, 3)
        final = evaluated["state"]
        require(final["evaluation"]["recommendation"], "evaluation recommendation is required")
        require(len(final["evidence"]) >= 3, "evaluation should have at least three evidence items")
        require(
            len({item["competency"] for item in final["evidence"] if item["competency"]}) >= 2,
            "evaluation should cover at least two competencies",
        )

        result = {
            "session_id": session_id,
            "complete": True,
            "timings": timings,
            "total_seconds": round(time.perf_counter() - started_at, 3),
            "checkpoints": checkpoints,
            "live_records": len(final["live_interview_records"]),
            "evidence_items": len(final["evidence"]),
            "recommendation": final["evaluation"]["recommendation"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
