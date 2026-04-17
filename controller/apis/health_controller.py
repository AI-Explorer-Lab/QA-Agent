from __future__ import annotations

from fastapi import APIRouter

from utils.config_loader import get_app_config

router = APIRouter()


@router.get("/health")
async def health():
    config = get_app_config()
    return {
        "status": "ok",
        "app": config.get("app", {}).get("name", "trusted-pdf-qa"),
        "storage_backend": config.get("storage", {}).get("backend", "pgvector"),
        "embedding_dim": config.get("embedding", {}).get("dimension", 1024),
    }
