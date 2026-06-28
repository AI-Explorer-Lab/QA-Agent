from __future__ import annotations

from fastapi import APIRouter

from service.session.session_service import get_session_service

router = APIRouter()


@router.get("/qa/sessions/{session_id}")
async def get_session(session_id: str):
    session = await get_session_service().get_session(session_id)
    if session is None:
        return {"session_id": session_id, "messages": [], "retrieval_traces": []}
    return session
