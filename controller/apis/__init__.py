from fastapi import APIRouter

from controller.apis.document_controller import router as document_router
from controller.apis.health_controller import router as health_router
from controller.apis.qa_controller import router as qa_router
from controller.apis.session_controller import router as session_router

router = APIRouter()
router.include_router(health_router)
router.include_router(document_router)
router.include_router(qa_router)
router.include_router(session_router)

__all__ = ["router"]
