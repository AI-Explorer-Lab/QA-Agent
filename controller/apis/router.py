from __future__ import annotations

from fastapi import APIRouter

from controller.apis.document_controller import router as document_router
from controller.apis.health_controller import router as health_router
from controller.apis.qa_controller import router as qa_router
from controller.apis.session_controller import router as session_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(document_router, tags=["documents"])
api_router.include_router(qa_router, tags=["qa"])
api_router.include_router(session_router, tags=["sessions"])
