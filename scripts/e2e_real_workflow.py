"""Run a repeatable real-HTTP candidate interview through final evaluation.

This deliberately uses synthetic candidate data. It is intended for local release
verification against the configured model provider, not as a unit-test substitute.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

RESUME = """E2E 测试候选人
5 年 Python 后端开发经验。负责过日均 2,000 万请求的订单服务，将 P95 延迟从
420ms 降至 180ms；使用 PostgreSQL、Redis、Kafka 和 Kubernetes。曾带领 4 人小组，
通过容量压测、灰度发布与可观测性建设，将高峰期故障率降低 60%。以上均为合成测试数据。
"""

JOB_DESCRIPTION = """高级 Python 平台工程师
岗位职责：设计和维护高并发 Python 服务；建设可观测性、容量规划与故障恢复能力；
参与架构评审并指导团队成员。
任职要求：5 年以上后端经验；熟悉 FastAPI、PostgreSQL、Redis、消息队列与 Kubernetes；
能够用数据解释架构权衡并推动跨团队交付。
团队背景：平台工程团队共 8 人，服务交易与数据产品，当前重点是降低延迟和提升可靠性。
"""


def request_json(
    client: httpx.Client, method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = client.request(method, path, json=payload)
    response.raise_for_status()
    return response.json()


def synthetic_answer(question: dict[str, Any], answer_number: int) -> str:
    competency = question.get("competency") or "综合能力"
    return (
        f"这是第 {answer_number} 条合成 E2E 回答，针对{competency}。"
        "背景是订单服务在促销高峰 P95 达到 420ms，并出现连接池耗尽。"
        "我的任务是在不影响交易正确性的前提下把 P95 降到 200ms 以下。"
        "我先用 tracing 和慢查询日志定位热点，再增加查询索引、拆分批量写入、"
        "为只读数据设置有界 Redis 缓存，并用压测验证连接池和降级阈值。"
        "上线采用 5%、20%、50%、100% 灰度，监控延迟、错误率与缓存命中率。"
        "最终 P95 降到 180ms，高峰错误率下降 60%；代价是缓存一致性复杂度增加，"
        "因此补充了版本号失效、回源限流和回滚演练。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-answers", type=int, default=20)
    args = parser.parse_args()

    started_at = time.perf_counter()
    timings: dict[str, float] = {}
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        session = request_json(
            client,
            "POST",
            "/api/interviews/sessions",
            {
                "candidate_name": "E2E 测试候选人",
                "job_title": "高级 Python 平台工程师",
                "company_name": "E2E Test Company",
            },
        )
        session_id = session["id"]

        phase_started = time.perf_counter()
        request_json(
            client,
            "POST",
            f"/api/autopilot/{session_id}/run",
            {
                "role": "candidate",
                "resume_text": RESUME,
                "job_description": JOB_DESCRIPTION,
                "company_name": "E2E Test Company",
                "company_context": "合成的本地回归测试公司，不进行公开网络研究。",
                "authorized_public_research": False,
            },
        )
        timings["preparation_seconds"] = round(time.perf_counter() - phase_started, 3)

        answers = 0
        phase_started = time.perf_counter()
        while answers < args.max_answers:
            mock = request_json(client, "GET", f"/api/mock-interviews/{session_id}")
            question = mock.get("current_question")
            if question is None:
                break
            answers += 1
            request_json(
                client,
                "POST",
                f"/api/mock-interviews/{session_id}/answers",
                {
                    "question_id": question["id"],
                    "answer": synthetic_answer(question, answers),
                },
            )
        else:
            raise RuntimeError(f"Interview exceeded safety limit of {args.max_answers} answers")
        timings["interview_and_evaluation_seconds"] = round(
            time.perf_counter() - phase_started, 3
        )

        final = request_json(client, "GET", f"/api/interviews/sessions/{session_id}")["state"]
        if final["mock_session"]["status"] != "completed":
            raise RuntimeError("Mock interview did not complete")
        if final["autopilot"]["status"] != "completed":
            raise RuntimeError("Autopilot did not produce the final evaluation")

        sources = sorted({item["source"] for item in final["evidence"]})
        result = {
            "session_id": session_id,
            "complete": True,
            "timings": timings,
            "total_seconds": round(time.perf_counter() - started_at, 3),
            "planned_questions": len(final["mock_interview"]["questions"]),
            "recorded_answers": len(final["mock_session"]["responses"]),
            "evidence_items": len(final["evidence"]),
            "evidence_sources": sources,
            "recommendation": final["evaluation"]["recommendation"],
            "feedback_evidence_safe": "证据不足" in final["feedback"]["overall"]
            if final["evaluation"]["recommendation"] == "insufficient_evidence"
            else True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
