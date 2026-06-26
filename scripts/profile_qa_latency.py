from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVENTS: list[dict[str, Any]] = []


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _safe_extra(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item not in (None, "")}


def _record(label: str, started: float, **extra: Any) -> None:
    EVENTS.append({"label": label, "duration_ms": _duration_ms(started), **_safe_extra(extra)})


def _patch_async(cls: type[Any], method_name: str, label: str, extra_fn: Callable[..., dict[str, Any]] | None = None) -> None:
    original = getattr(cls, method_name)

    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original(self, *args, **kwargs)
        finally:
            extra = extra_fn(self, args, kwargs) if extra_fn else {}
            _record(label, started, **extra)

    setattr(cls, method_name, wrapper)


def _patch_sync(cls: type[Any], method_name: str, label: str, extra_fn: Callable[..., dict[str, Any]] | None = None) -> None:
    original = getattr(cls, method_name)

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            extra = extra_fn(self, args, kwargs) if extra_fn else {}
            _record(label, started, **extra)

    setattr(cls, method_name, wrapper)


def install_patches() -> None:
    from core.config_loader import load_runtime_env

    load_runtime_env()

    from service.agent.answer_generator import AnswerGenerator
    from service.agent.controlled_agents import IntentUnderstandingAgent, SlotFillingAgent
    from service.agent.conversation_context import ConversationContextService
    from service.agent.evidence_gate import EvidenceDecisionEngine
    from service.agent.trusted_qa_workflow import TrustedQAWorkflow
    from service.embedding.embedding_service import EmbeddingService
    from service.llm.llm_client import LLMService
    from service.retrieval.hybrid_retriever import HybridRetriever
    from service.retrieval.parallel_query_executor import ParallelQueryExecutor
    from service.retrieval.two_stage_hybrid_reranker import TwoStageHybridReranker

    _patch_async(ConversationContextService, "prepare_context", "conversation.prepare_context")
    _patch_async(IntentUnderstandingAgent, "classify", "agent.intent_classify")
    _patch_async(SlotFillingAgent, "fill", "agent.slot_fill")
    _patch_async(TrustedQAWorkflow, "_retrieve_with_cache_aware_expansion", "workflow.retrieve_with_expansion")
    _patch_async(EvidenceDecisionEngine, "evaluate", "evidence.evaluate")
    _patch_async(LLMService, "complete", "llm.complete", lambda self, _a, _k: {"mode": getattr(self, "last_call_mode", ""), "model": getattr(self, "model", "")})
    _patch_async(LLMService, "structured_json", "llm.structured_json", lambda self, _a, _k: {"mode": getattr(self, "last_call_mode", ""), "model": getattr(self, "model", "")})
    _patch_async(LLMService, "generate_grounded_answer", "llm.generate_grounded_answer", lambda self, _a, _k: {"mode": getattr(self, "last_call_mode", ""), "model": getattr(self, "model", "")})
    _patch_async(EmbeddingService, "embed_text", "embedding.embed_text", lambda self, args, _k: {"provider": getattr(self, "provider_name", ""), "chars": len(str(args[0] if args else ""))})
    _patch_async(HybridRetriever, "retrieve", "retrieval.hybrid_retrieve")
    _patch_async(ParallelQueryExecutor, "execute", "retrieval.parallel_execute")
    _patch_async(ParallelQueryExecutor, "_run_route_task", "retrieval.route_task", lambda _self, args, _k: {"route": args[0] if args else ""})
    _patch_sync(TwoStageHybridReranker, "rerank", "rerank.two_stage")
    _patch_sync(AnswerGenerator, "generate", "answer.rule_generate")


def summarize_events() -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for event in EVENTS:
        grouped[str(event.get("label"))].append(int(event.get("duration_ms") or 0))

    summary = []
    for label, values in sorted(grouped.items(), key=lambda item: sum(item[1]), reverse=True):
        summary.append(
            {
                "label": label,
                "count": len(values),
                "sum_ms": sum(values),
                "max_ms": max(values),
                "avg_ms": round(statistics.mean(values), 1),
            }
        )
    return {"summary": summary, "events": [dict(event) for event in EVENTS]}


def _compact_response(data: dict[str, Any]) -> dict[str, Any]:
    trace = data.get("retrieval_trace") or {}
    rerank = data.get("rerank_trace") or {}
    return {
        "decision": data.get("decision"),
        "query_type": data.get("query_type"),
        "confidence": data.get("confidence"),
        "answer_chars": len(str(data.get("answer") or "")),
        "citations": len(data.get("citations") or []),
        "evidence": len(data.get("evidence") or []),
        "workflow_runner": trace.get("workflow_runner"),
        "llm": trace.get("llm"),
        "progress_stages": trace.get("progress_stages"),
        "task_trace": trace.get("task_trace"),
        "rerank_cross_encoder": rerank.get("cross_encoder") if isinstance(rerank, dict) else {},
        "rerank_input_candidates": rerank.get("input_candidates") if isinstance(rerank, dict) else None,
    }


async def run_once(args: argparse.Namespace) -> dict[str, Any]:
    import httpx
    from main import app
    if getattr(args, "disable_retry", False):
        from service.agent.trusted_qa_workflow import get_trusted_qa_workflow

        workflow = get_trusted_qa_workflow()
        workflow.evidence_decision.retry_limit = 0
        workflow.evidence_decision.rule_gate.retry_limit = 0

    payload = {
        "question": args.question,
        "collection_name": args.collection,
        "top_k": args.top_k,
        "expand_query_num": args.expand_query_num,
        "enable_cache": args.enable_cache,
        "include_debug": True,
    }
    transport = httpx.ASGITransport(app=app)
    started = time.perf_counter()
    async with httpx.AsyncClient(transport=transport, base_url="http://trusted-qa.local", timeout=args.timeout) as client:
        response = await client.post("/qa/ask", json=payload, timeout=args.timeout)
    total_ms = _duration_ms(started)
    data = response.json()
    return {
        "request": payload,
        "status_code": response.status_code,
        "total_ms": total_ms,
        "response": _compact_response(data if isinstance(data, dict) else {}),
        "profile": summarize_events(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the /qa/ask API path in-process through FastAPI ASGI.")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--question", default="What is the company's 2025 operating revenue?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expand-query-num", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--disable-retry", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    args.enable_cache = not args.no_cache
    install_patches()
    results = []
    for index in range(max(1, int(args.runs))):
        EVENTS.clear()
        result = await run_once(args)
        result["run_index"] = index + 1
        results.append(result)
    payload = results[0] if len(results) == 1 else {"runs": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())




