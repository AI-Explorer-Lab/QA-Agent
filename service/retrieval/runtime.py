from __future__ import annotations

import threading
from typing import Any, Iterable, Mapping

from database.connection import get_pgvector_database_url, get_storage_backend
from service.retrieval.pgvector_repository import PgvectorRepository
from utils.config_loader import get_app_config

_LOCK = threading.Lock()


def _configured_embedding_dim(config: Mapping[str, Any]) -> int:
    storage = config.get("storage", {}) if isinstance(config.get("storage"), Mapping) else {}
    pgvector = storage.get("pgvector", {}) if isinstance(storage.get("pgvector"), Mapping) else {}
    return int(pgvector.get("embedding_dim") or 1024)


def _build_runtime_repository() -> PgvectorRepository:
    config = get_app_config()
    backend = get_storage_backend(config).strip().lower() or "pgvector"
    if backend != "pgvector":
        raise RuntimeError("Runtime repository must use pgvector. Set storage.backend=pgvector in config/app.yaml.")

    database_url = get_pgvector_database_url(config)
    return PgvectorRepository(
        backend="pgvector",
        database_url=database_url,
        embedding_dim=_configured_embedding_dim(config),
    )


_RUNTIME_REPOSITORY = _build_runtime_repository()


def get_runtime_repository() -> PgvectorRepository:
    return _RUNTIME_REPOSITORY


def reset_runtime_repository() -> None:
    raise RuntimeError("reset_runtime_repository is disabled because runtime storage is pgvector-only.")


def replace_collection_chunks(collection_name: str, chunks: Iterable[Mapping[str, Any]]) -> int:
    rows = [dict(item) for item in chunks]
    with _LOCK:
        return _RUNTIME_REPOSITORY.replace_collection_chunks(collection_name, rows)


def upsert_runtime_chunks(chunks: Iterable[Mapping[str, Any]]) -> int:
    rows = [dict(item) for item in chunks]
    with _LOCK:
        return _RUNTIME_REPOSITORY.upsert_chunks(rows)
