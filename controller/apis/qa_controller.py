from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from domain.qa import QARequest
from exception import CollectionNotFoundException, ValidationException
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
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "repository_collection_count": retrieval_trace.get("repository_collection_count", 0),
            "workflow_runner": retrieval_trace.get("workflow_runner", ""),
            "progress_stages": retrieval_trace.get("progress_stages", []),
        },
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _validate_qa_request(request: QARequest) -> str:
    collection_name = str(request.collection_name or "").strip()
    if not collection_name:
        raise ValidationException("collection_name is required", detail={"collection_name": request.collection_name})
    if get_runtime_repository().count_collection_chunks(collection_name) <= 0:
        raise CollectionNotFoundException(collection_name)
    return collection_name


async def _run_qa(request: QARequest, collection_name: str) -> dict[str, Any]:
    return await get_trusted_qa_workflow().ask(
        question=request.question,
        session_id=request.session_id or None,
        collection_name=collection_name,
        top_k=request.top_k,
        expand_query_num=request.expand_query_num,
        enable_cache=request.enable_cache,
    )


@router.post("/qa/ask")
async def ask(request: QARequest):
    collection_name = _validate_qa_request(request)
    response = await _run_qa(request, collection_name)
    if request.include_debug:
        return response
    return _compact_qa_response(response)


@router.post("/qa/ask/stream")
async def ask_stream(request: QARequest):
    collection_name = _validate_qa_request(request)

    async def event_stream():
        yield _sse_event(
            "status",
            {
                "message": STREAM_WAITING_MESSAGE,
                "stage": "started",
                "collection_name": collection_name,
            },
        )
        task = asyncio.create_task(_run_qa(request, collection_name))
        stage_plan = [
            ("understanding", "?????????????"),
            ("retrieving", "?????????"),
            ("answering", "????????"),
        ]
        stage_index = 0
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=2)
                if done:
                    break
                stage, message = stage_plan[min(stage_index, len(stage_plan) - 1)]
                stage_index += 1
                yield _sse_event(
                    "status",
                    {
                        "message": message,
                        "stage": stage,
                        "collection_name": collection_name,
                    },
                )
            response = await task
            payload = response if request.include_debug else _compact_qa_response(response)
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
