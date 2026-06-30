from __future__ import annotations

from fastapi import APIRouter

from service.session.session_service import get_session_service

router = APIRouter()


@router.get("/qa/sessions")
async def list_sessions(collection_name: str = "default", limit: int = 30, offset: int = 0):
    return await get_session_service().list_sessions(collection_name=collection_name, limit=limit, offset=offset)


@router.get("/qa/sessions/{session_id}")
async def get_session(session_id: str):
    session = await get_session_service().get_session(session_id)
    if session is None:
        return {"session_id": session_id, "messages": [], "retrieval_traces": []}
    return session


@router.delete("/qa/sessions/{session_id}")
async def delete_session(session_id: str):
    return await get_session_service().delete_session(session_id)
