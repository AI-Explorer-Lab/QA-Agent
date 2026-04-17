from __future__ import annotations

import threading
from typing import Any, Iterable, Mapping

from service.retrieval.pgvector_repository import PgvectorRepository

_LOCK = threading.Lock()
_RUNTIME_REPOSITORY = PgvectorRepository(backend="local_dev", embedding_dim=1024)


def get_runtime_repository() -> PgvectorRepository:
    return _RUNTIME_REPOSITORY


def reset_runtime_repository() -> None:
    with _LOCK:
        _RUNTIME_REPOSITORY._local_chunks = []
        _RUNTIME_REPOSITORY._sparse_retriever.index_chunks([])


def replace_collection_chunks(collection_name: str, chunks: Iterable[Mapping[str, Any]]) -> int:
    rows = [dict(item) for item in chunks]
    with _LOCK:
        _RUNTIME_REPOSITORY._local_chunks = [
            row for row in _RUNTIME_REPOSITORY._local_chunks
            if str(row.get("collection_name") or "") != collection_name
        ]
        return _RUNTIME_REPOSITORY.upsert_chunks(rows)


def upsert_runtime_chunks(chunks: Iterable[Mapping[str, Any]]) -> int:
    rows = [dict(item) for item in chunks]
    with _LOCK:
        return _RUNTIME_REPOSITORY.upsert_chunks(rows)
