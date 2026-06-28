"""Retrieval trace mapper with pgvector and local_dev backend support."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone

from sqlalchemy import text

from database import get_async_session, get_storage_backend
from domain import RetrievalTrace
from mapper.local_dev_repository import LocalDevRepository


class RetrievalTraceMapper:
    def __init__(
        self,
        backend: str | None = None,
        database_url: str | None = None,
        local_repository: LocalDevRepository | None = None,
    ) -> None:
        self.backend = (backend or get_storage_backend()).strip()
        self.local_repository = local_repository

        if self.backend == "local_dev":
            self.local_repository = local_repository or LocalDevRepository(database_url)
            self.database_url = database_url
        else:
            self.database_url = database_url

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _to_json(data: object) -> str:
        return json.dumps(data, ensure_ascii=True, default=str)

    async def save_trace(self, trace: RetrievalTrace) -> None:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            await asyncio.to_thread(self.local_repository.save_retrieval_trace, trace)
            return

        payload = trace.model_dump(mode="json")
        retrieval_trace_payload = {
            "dense_candidates": payload.get("dense_candidates", []),
            "sparse_candidates": payload.get("sparse_candidates", []),
            "merged_candidates": payload.get("merged_candidates", []),
            "latency_ms": payload.get("latency_ms", 0.0),
        }

        statement = text(
            """
            INSERT INTO retrieval_traces (
                trace_id,
                session_id,
                message_id,
                collection_name,
                question,
                expanded_queries_json,
                retrieval_trace_json,
                rerank_trace_json,
                selected_candidates_json,
                created_at
            ) VALUES (
                :trace_id,
                :session_id,
                :message_id,
                :collection_name,
                :question,
                CAST(:expanded_queries_json AS JSONB),
                CAST(:retrieval_trace_json AS JSONB),
                CAST(:rerank_trace_json AS JSONB),
                CAST(:selected_candidates_json AS JSONB),
                :created_at
            )
            ON CONFLICT (trace_id) DO UPDATE SET
                session_id=EXCLUDED.session_id,
                message_id=EXCLUDED.message_id,
                collection_name=EXCLUDED.collection_name,
                question=EXCLUDED.question,
                expanded_queries_json=EXCLUDED.expanded_queries_json,
                retrieval_trace_json=EXCLUDED.retrieval_trace_json,
                rerank_trace_json=EXCLUDED.rerank_trace_json,
                selected_candidates_json=EXCLUDED.selected_candidates_json,
                created_at=EXCLUDED.created_at
            """
        )

        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            await session.execute(
                statement,
                {
                    "trace_id": payload["trace_id"],
                    "session_id": payload.get("session_id", ""),
                    "message_id": payload.get("message_id", ""),
                    "collection_name": payload.get("collection_name", ""),
                    "question": payload.get("question", ""),
                    "expanded_queries_json": self._to_json(payload.get("expanded_queries", [])),
                    "retrieval_trace_json": self._to_json(retrieval_trace_payload),
                    "rerank_trace_json": self._to_json(payload.get("rerank_trace", {})),
                    "selected_candidates_json": self._to_json(payload.get("selected_candidates", [])),
                    "created_at": payload.get("created_at") or self._now_utc(),
                },
            )
            await session.commit()

    async def list_session_traces(self, session_id: str, limit: int = 100) -> list[RetrievalTrace]:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            return await asyncio.to_thread(self.local_repository.list_retrieval_traces, session_id, limit)

        query = text(
            """
            SELECT
                trace_id,
                session_id,
                message_id,
                collection_name,
                question,
                expanded_queries_json,
                retrieval_trace_json,
                rerank_trace_json,
                selected_candidates_json,
                created_at
            FROM retrieval_traces
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )

        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            rows = (
                await session.execute(
                query,
                {"session_id": session_id, "limit": max(1, int(limit))},
                )
            ).mappings().all()

        traces: list[RetrievalTrace] = []
        for row in rows:
            retrieval_payload = row["retrieval_trace_json"] or {}
            traces.append(
                RetrievalTrace(
                    trace_id=row["trace_id"],
                    session_id=row["session_id"],
                    message_id=row["message_id"],
                    collection_name=row["collection_name"],
                    question=row["question"],
                    expanded_queries=row["expanded_queries_json"] or [],
                    dense_candidates=retrieval_payload.get("dense_candidates", []),
                    sparse_candidates=retrieval_payload.get("sparse_candidates", []),
                    merged_candidates=retrieval_payload.get("merged_candidates", []),
                    selected_candidates=row["selected_candidates_json"] or [],
                    rerank_trace=row["rerank_trace_json"] or {},
                    latency_ms=float(retrieval_payload.get("latency_ms", 0.0) or 0.0),
                    created_at=row["created_at"],
                )
            )
        return traces
