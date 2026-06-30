from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.agent.conversation_context import ConversationContextService
from service.agent.trusted_qa_workflow import get_trusted_qa_workflow


EXPECTED_QUERY_TYPES = {
    "text": "fact_lookup",
    "table": "table_qa",
    "skill_fact": "fact_lookup",
    "skill_table": "table_qa",
    "citation": "citation_locate",
    "summary": "summarization",
    "report": "report_generation",
}


def _split_markdown_row(line: str) -> List[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _case_kind(row_id: str) -> str:
    for prefix in sorted(EXPECTED_QUERY_TYPES, key=len, reverse=True):
        if row_id.startswith(f"{prefix}_"):
            return prefix
    if row_id.startswith("route_"):
        return "route"
    return ""


def _parse_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        parts = _split_markdown_row(line)
        if not parts:
            continue
        row_id = parts[0]
        kind = _case_kind(row_id)
        if not kind:
            continue
        if kind == "route":
            cases.append(
                {
                    "id": row_id,
                    "kind": kind,
                    "question": parts[3],
                    "expected_route": parts[4],
                    "expected_context_source": parts[5],
                    "expected_effective_question": parts[6],
                    "expected_history_refs": [item.strip() for item in parts[7].split(",") if item.strip()],
                    "keywords": [],
                }
            )
        else:
            cases.append(
                {
                    "id": row_id,
                    "kind": kind,
                    "question": parts[1],
                    "expected_query_type": EXPECTED_QUERY_TYPES[kind],
                    "keywords": [item.strip() for item in parts[3].split("、") if item.strip()],
                }
            )
    return cases


def _evidence_text(evidence: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for key in (
            "content",
            "raw_doc",
            "heading_path",
            "table_header_text",
            "table_context_text",
            "doc_source",
        ):
            value = item.get(key)
            if value not in (None, ""):
                parts.append(str(value))
        for key in ("heading_path", "page_idx", "table_id"):
            value = metadata.get(key)
            if value not in (None, ""):
                parts.append(str(value))
    return "\n".join(parts)


def _keyword_recall(keywords: List[str], evidence: List[Dict[str, Any]]) -> tuple[float, int, int]:
    if not keywords:
        return 0.0, 0, 0
    haystack = _evidence_text(evidence)
    matched = sum(1 for keyword in keywords if keyword and keyword in haystack)
    return matched / max(1, len(keywords)), matched, len(keywords)


class _FakeTurnRouterLLM:
    async def structured_json(self, system_prompt, user_payload, schema, max_tokens=700):
        state = user_payload["conversation_state"]
        question = state["current_question"]
        effective_by_question = {
            "2024年这个指标呢？": "2024年芯导科技营业收入是多少？",
            "占营收比例也给一下": "2025年芯导科技研发投入占营业收入比例是多少？",
            "上一条出处在哪里？": "定位上一轮回答中前五名客户销售额占年度销售总额51.39%的出处和页码。",
            "上一条原文引用给我。": "定位上一轮回答中公司2025年度未签署远期外汇合约或货币互换合约的原文引用。",
        }
        if question in {"2025年", "2025年年度报告"}:
            turn_type = "clarification_reply"
            effective_question = question
        elif question in {"上一条出处在哪里？", "上一条原文引用给我。"}:
            turn_type = "citation_followup"
            effective_question = effective_by_question[question]
        else:
            turn_type = "follow_up"
            effective_question = effective_by_question[question]
        return {
            "turn_type": turn_type,
            "should_use_history": False,
            "context_source": "none",
            "effective_question": effective_question,
            "history_refs": [],
            "missing_info": [],
            "confidence": 0.93,
            "reason": "deterministic route-turn evaluation double",
        }


def _completed_turn_session() -> Dict[str, Any]:
    return {
        "collection_name": "xindao",
        "metadata": {
            "conversation_focus": {
                "active_topic": "2025年芯导科技营业收入是多少？",
                "company": "芯导科技",
                "period": "2025年",
                "metric": "营业收入",
                "last_query_type": "table_qa",
                "updated_at_turn": "turn_1",
            }
        },
        "messages": [
            {"role": "user", "content": "2025年芯导科技营业收入是多少？"},
            {
                "role": "assistant",
                "content": "2025年营业收入为393,607,502.95元。",
                "metadata": {
                    "decision": "answer",
                    "query_type": "table_qa",
                    "effective_question": "2025年芯导科技营业收入是多少？",
                    "skill_trace": {"slots": {"company": "芯导科技", "period": "2025年", "metric": "营业收入"}},
                    "evidence": [{"doc_source": "annual_report_2025.pdf", "content": "营业收入393,607,502.95元"}],
                    "citations": [{"citation_id": "C1", "doc_source": "annual_report_2025.pdf"}],
                },
            },
        ],
    }


def _long_history_session() -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    for index in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"历史问题{index}是什么？"},
                {
                    "role": "assistant",
                    "content": f"历史回答{index}。",
                    "metadata": {
                        "decision": "answer",
                        "query_type": "fact_lookup",
                        "effective_question": f"历史问题{index}是什么？",
                        "evidence": [{"doc_source": "annual_report_2025.pdf", "content": f"历史证据{index}"}],
                        "citations": [{"citation_id": f"C{index}", "doc_source": "annual_report_2025.pdf"}],
                    },
                },
            ]
        )
    return {
        "collection_name": "xindao",
        "metadata": {
            "conversation_focus": {
                "active_topic": "2025年芯导科技研发投入是多少？",
                "company": "芯导科技",
                "period": "2025年",
                "metric": "研发投入",
                "last_query_type": "table_qa",
                "updated_at_turn": "turn_8",
            }
        },
        "messages": messages,
    }


def _clarification_session(original_question: str, metric: str) -> Dict[str, Any]:
    return {
        "collection_name": "xindao",
        "messages": [
            {"role": "user", "content": original_question},
            {
                "role": "assistant",
                "content": "请补充要查询的期间。",
                "metadata": {
                    "decision": "clarify",
                    "query_type": "table_qa",
                    "effective_question": original_question,
                    "skill_trace": {"slots": {"metric": metric, "__missing_required__": ["period"]}},
                },
            },
        ],
    }


def _citation_session(claim: str) -> Dict[str, Any]:
    return {
        "collection_name": "xindao",
        "messages": [
            {"role": "user", "content": "请回答并给出依据。"},
            {
                "role": "assistant",
                "content": claim,
                "metadata": {
                    "decision": "answer",
                    "query_type": "fact_lookup",
                    "effective_question": "请回答并给出依据。",
                    "evidence": [{"doc_source": "annual_report_2025.pdf", "content": claim}],
                    "citations": [{"citation_id": "C1", "doc_source": "annual_report_2025.pdf"}],
                },
            },
        ],
    }


def _session_for_route_case(row_id: str) -> Dict[str, Any]:
    if row_id == "route_follow_001":
        return _completed_turn_session()
    if row_id == "route_follow_002":
        return _long_history_session()
    if row_id == "route_clarify_001":
        return _clarification_session("营业收入是多少？", "营业收入")
    if row_id == "route_clarify_002":
        return _clarification_session("研发投入是多少？", "研发投入")
    if row_id == "route_citation_001":
        return _citation_session("前五名客户销售额占年度销售总额51.39%。")
    if row_id == "route_citation_002":
        return _citation_session("公司2025年度未签署远期外汇合约或货币互换合约。")
    raise ValueError(f"Unknown route case: {row_id}")


async def _evaluate_route_case(case: Dict[str, Any], service: ConversationContextService) -> Dict[str, Any]:
    started = time.perf_counter()
    context = await service.prepare_context(_session_for_route_case(case["id"]), case["question"], "xindao")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    route = context["turn_route"]
    route_correct = (
        route["turn_type"] == case["expected_route"]
        and route["context_source"] == case["expected_context_source"]
        and route["history_refs"] == case["expected_history_refs"]
        and context["effective_question"] == case["expected_effective_question"]
    )
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_route": case["expected_route"],
        "actual_route": route["turn_type"],
        "route_correct": route_correct,
        "elapsed_ms": elapsed_ms,
        "topk_recall": None,
        "faithfulness": None,
        "decision": "",
        "evidence_count": None,
        "matched_keywords": None,
        "keyword_count": None,
    }


async def _evaluate_qa_case(case: Dict[str, Any], workflow: Any, collection_name: str, top_k: int) -> Dict[str, Any]:
    started = time.perf_counter()
    result = await workflow.ask(
        question=case["question"],
        collection_name=collection_name,
        session_id=f"eval-{case['id']}-{uuid4().hex}",
        top_k=top_k,
        enable_cache=True,
        use_llm_intent_slot=False,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    trace = result.get("retrieval_trace") or {}
    evaluation = trace.get("evaluation") or {}
    evidence = list(result.get("evidence") or [])
    recall, matched, total = _keyword_recall(case["keywords"], evidence[:top_k])
    actual_query_type = str(result.get("query_type") or "")
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_route": case["expected_query_type"],
        "actual_route": actual_query_type,
        "route_correct": actual_query_type == case["expected_query_type"],
        "elapsed_ms": elapsed_ms,
        "topk_recall": round(recall, 4),
        "faithfulness": evaluation.get("grounding_score"),
        "overall_score": evaluation.get("overall_score"),
        "decision": result.get("decision"),
        "evidence_count": len(evidence),
        "matched_keywords": matched,
        "keyword_count": total,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "OK" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _to_markdown(results: List[Dict[str, Any]]) -> str:
    headers = ["id", "路由", "耗时ms", "topk召回率", "忠实度", "证据关键词", "决策"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---|---:|---:|---:|---|---|"]
    for row in results:
        keyword_cell = "N/A"
        if row.get("keyword_count") is not None:
            keyword_cell = f"{row.get('matched_keywords')}/{row.get('keyword_count')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["id"]),
                    _fmt(row["route_correct"]) + f" ({_fmt(row['actual_route'])})",
                    _fmt(row["elapsed_ms"]),
                    _fmt(row["topk_recall"]),
                    _fmt(row["faithfulness"]),
                    keyword_cell,
                    _fmt(row.get("decision", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="xindao")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data", default="tests/test_data.md")
    parser.add_argument("--out-json", default="test_data_eval_results.json")
    parser.add_argument("--out-md", default="test_data_eval_results.md")
    args = parser.parse_args()

    os.environ.setdefault("TRUSTED_QA_ENABLE_REAL_LLM", "0")
    cases = _parse_cases(Path(args.data))
    workflow = get_trusted_qa_workflow()
    route_service = ConversationContextService(_FakeTurnRouterLLM())
    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['question']}", flush=True)
        if case["kind"] == "route":
            row = await _evaluate_route_case(case, route_service)
        else:
            row = await _evaluate_qa_case(case, workflow, args.collection, args.top_k)
        results.append(row)
        print(
            f"  route={row['actual_route']} ok={row['route_correct']} "
            f"ms={row['elapsed_ms']} recall={row['topk_recall']} faith={row['faithfulness']}",
            flush=True,
        )

    summary = {
        "collection": args.collection,
        "top_k": args.top_k,
        "case_count": len(results),
        "route_accuracy": sum(1 for row in results if row["route_correct"]) / max(1, len(results)),
        "avg_elapsed_ms": sum(float(row["elapsed_ms"]) for row in results) / max(1, len(results)),
        "avg_topk_recall": sum(float(row["topk_recall"]) for row in results if row["topk_recall"] is not None)
        / max(1, sum(1 for row in results if row["topk_recall"] is not None)),
        "avg_faithfulness": sum(float(row["faithfulness"]) for row in results if row["faithfulness"] is not None)
        / max(1, sum(1 for row in results if row["faithfulness"] is not None)),
    }
    payload = {"summary": summary, "results": results}
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = _to_markdown(results)
    Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
