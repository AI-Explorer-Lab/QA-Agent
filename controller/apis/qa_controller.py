from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from domain.qa import QARequest
from exceptions import CollectionNotFoundException, ValidationException
from service.agent.trusted_qa_workflow import get_trusted_qa_workflow
from service.retrieval.runtime import get_runtime_repository

router = APIRouter()
STREAM_WAITING_MESSAGE = "正在生成答案，请稍等......"


def _source_name(doc_source: Any) -> str:
    source = str(doc_source or "").strip()
    if not source:
        return ""
    try:
        return Path(source).name or source
    except Exception:
        return source.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]


def _compact_citation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_id": item.get("citation_id", ""),
        "chunk_id": item.get("chunk_id", ""),
        "doc_id": item.get("doc_id", ""),
        "source_name": _source_name(item.get("doc_source")),
        "page_idx": item.get("page_idx"),
        "page_range": item.get("page_range", ""),
        "heading_path": item.get("heading_path", ""),
        "quote": item.get("quote", ""),
        "confidence": item.get("confidence", 0),
    }


def _compact_qa_response(response: dict[str, Any]) -> dict[str, Any]:
    retrieval_trace = response.get("retrieval_trace") or {}
    citations = response.get("citations") or []
    evidence = response.get("evidence") or []
    return {
        "answer": response.get("answer", ""),
        "decision": response.get("decision", ""),
        "query_type": response.get("query_type", ""),
        "confidence": response.get("confidence", 0),
        "session_id": response.get("session_id", ""),
        "citations": [_compact_citation(item) for item in citations if isinstance(item, dict)],
        "retrieval": {
            "collection_name": retrieval_trace.get("collection_name", ""),
            "trace_id": retrieval_trace.get("trace_id", ""),
            "cache_hit": bool(retrieval_trace.get("cache_hit", False)),
            "query_expansion_cache_hit": bool(retrieval_trace.get("query_expansion_cache_hit", False)),
            "query_expansion_skipped": retrieval_trace.get("query_expansion_skipped", ""),
            "llm_query_expansion_used": bool(retrieval_trace.get("llm_query_expansion_used", False)),
            "llm_answer_cache_hit": bool((retrieval_trace.get("llm") or {}).get("answer_cache_hit", False)),
            "final_response_cache_hit": bool(retrieval_trace.get("final_response_cache_hit", False)),
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "repository_collection_count": retrieval_trace.get("repository_collection_count", 0),
            "workflow_runner": retrieval_trace.get("workflow_runner", ""),
            "workflow_duration_ms": retrieval_trace.get("workflow_duration_ms", 0),
            "progress_stages": retrieval_trace.get("progress_stages", []),
        },
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _stage_message(stage: str, status: str = "completed") -> str:
    running_messages = {
        "load_session": "正在加载会话",
        "conversation_context": "正在整理上下文",
        "intent_slot_understanding_agent": "正在识别意图槽位",
        "select_skill_from_registry": "正在进行技能路由",
        "clarify_gate": "正在判断是否需要澄清",
        "parallel_hybrid_retrieval": "正在检索证据",
        "retry_retrieval": "正在重试检索",
        "evidence_decision": "正在校验证据",
        "answer_generation": "正在生成回答",
        "finalize_response": "正在整理响应",
    }
    completed_messages = {
        "load_session": "会话加载完成",
        "conversation_context": "上下文整理完成",
        "intent_slot_understanding_agent": "意图槽位识别完成",
        "select_skill_from_registry": "技能路由完成",
        "clarify_gate": "澄清判断完成",
        "parallel_hybrid_retrieval": "证据检索完成",
        "retry_retrieval": "重试检索完成",
        "evidence_decision": "证据校验完成",
        "answer_generation": "回答生成完成",
        "finalize_response": "响应整理完成",
    }
    if status == "running":
        return running_messages.get(stage, stage or STREAM_WAITING_MESSAGE)
    return completed_messages.get(stage, stage or STREAM_WAITING_MESSAGE)


async def _validate_qa_request(request: QARequest) -> str:
    collection_name = str(request.collection_name or "").strip()
    if not collection_name:
        raise ValidationException("collection_name is required", detail={"collection_name": request.collection_name})
    if await get_runtime_repository().count_collection_chunks(collection_name) <= 0:
        raise CollectionNotFoundException(collection_name)
    return collection_name


async def _run_qa(request: QARequest, collection_name: str, progress_callback=None) -> dict[str, Any]:
    return await get_trusted_qa_workflow().ask(
        question=request.question,
        session_id=request.session_id or None,
        collection_name=collection_name,
        top_k=request.top_k,
        expand_query_num=request.expand_query_num,
        enable_cache=request.enable_cache,
        progress_callback=progress_callback,
    )


@router.post("/qa/ask")
async def ask(request: QARequest):
    collection_name = await _validate_qa_request(request)
    response = await _run_qa(request, collection_name)
    if request.include_debug:
        return response
    return _compact_qa_response(response)


@router.post("/qa/ask/stream")
async def ask_stream(request: QARequest):
    collection_name = await _validate_qa_request(request)

    async def event_stream():
        started_at = time.perf_counter()
        progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started_at) * 1000)

        async def progress_callback(stage: dict[str, Any]) -> None:
            await progress_queue.put(stage)

        task = asyncio.create_task(_run_qa(request, collection_name, progress_callback=progress_callback))
        try:
            while True:
                if task.done() and progress_queue.empty():
                    break
                try:
                    stage_event = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                stage = str(stage_event.get("stage") or stage_event.get("phase") or "")
                status = str(stage_event.get("status") or "completed")
                yield _sse_event(
                    "status",
                    {
                        **stage_event,
                        "message": _stage_message(stage, status),
                        "stage": stage,
                        "status": status,
                        "collection_name": collection_name,
                        "elapsed_ms": elapsed_ms(),
                    },
                )
            response = await task
            payload = response if request.include_debug else _compact_qa_response(response)
            yield _sse_event(
                "status",
                {
                    "message": "答案生成完成，正在输出",
                    "stage": "answer_generation",
                    "status": "completed",
                    "collection_name": collection_name,
                    "elapsed_ms": elapsed_ms(),
                },
            )
            yield _sse_event("final", payload)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

