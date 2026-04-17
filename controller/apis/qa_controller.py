from __future__ import annotations

from fastapi import APIRouter

from domain.qa import QARequest
from service.agent.trusted_qa_workflow import get_trusted_qa_workflow

router = APIRouter()


@router.post("/qa/ask")
async def ask(request: QARequest):
    return await get_trusted_qa_workflow().ask(
        question=request.question,
        session_id=request.session_id or None,
        collection_name=request.collection_name,
        top_k=request.top_k,
        expand_query_num=request.expand_query_num,
        enable_cache=request.enable_cache,
    )
