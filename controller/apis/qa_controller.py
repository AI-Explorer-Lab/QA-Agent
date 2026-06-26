from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from domain.qa import QARequest
from exception import CollectionNotFoundException, ValidationException
from service.agent.trusted_qa_workflow import get_trusted_qa_workflow
from service.retrieval.runtime import get_runtime_repository

router = APIRouter()


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


@router.post("/qa/ask")
async def ask(request: QARequest):
    collection_name = str(request.collection_name or "").strip()
    if not collection_name:
        raise ValidationException("collection_name is required", detail={"collection_name": request.collection_name})
    if get_runtime_repository().count_collection_chunks(collection_name) <= 0:
        raise CollectionNotFoundException(collection_name)
    response = await get_trusted_qa_workflow().ask(
        question=request.question,
        session_id=request.session_id or None,
        collection_name=collection_name,
        top_k=request.top_k,
        expand_query_num=request.expand_query_num,
        enable_cache=request.enable_cache,
    )
    if request.include_debug:
        return response
    return _compact_qa_response(response)
