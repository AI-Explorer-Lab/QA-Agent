"""Session/message mapper with pgvector and local_dev backend support."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone

from sqlalchemy import text

from database import get_async_session, get_storage_backend
from domain import QAMessage, QASession
from mapper.local_dev_repository import LocalDevRepository


class SessionMapper:
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

    async def save_session(self, session: QASession) -> None:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            await asyncio.to_thread(self.local_repository.save_session, session)
            return

        payload = session.model_dump(mode="json")
        statement = text(
            """
            INSERT INTO qa_sessions (
                session_id,
                collection_name,
                user_id,
                metadata_json,
                created_at,
                updated_at
            ) VALUES (
                :session_id,
                :collection_name,
                :user_id,
                CAST(:metadata_json AS JSONB),
                :created_at,
                :updated_at
            )
            ON CONFLICT (session_id) DO UPDATE SET
                collection_name=EXCLUDED.collection_name,
                user_id=EXCLUDED.user_id,
                metadata_json=EXCLUDED.metadata_json,
                updated_at=EXCLUDED.updated_at
            """
        )

        now_utc = self._now_utc()
        async with get_async_session(backend="pgvector", database_url=self.database_url) as db_session:
            await db_session.execute(
                statement,
                {
                    "session_id": payload["session_id"],
                    "collection_name": payload["collection_name"],
                    "user_id": payload.get("user_id", ""),
                    "metadata_json": self._to_json(payload.get("metadata", {})),
                    "created_at": payload.get("created_at") or now_utc,
                    "updated_at": payload.get("updated_at") or now_utc,
                },
            )
            await db_session.commit()

    async def get_session(self, session_id: str) -> QASession | None:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            return await asyncio.to_thread(self.local_repository.get_session, session_id)

        query = text(
            """
            SELECT
                session_id,
                collection_name,
                user_id,
                metadata_json,
                created_at,
                updated_at
            FROM qa_sessions
            WHERE session_id = :session_id
            LIMIT 1
            """
        )

        async with get_async_session(backend="pgvector", database_url=self.database_url) as db_session:
            row = (await db_session.execute(query, {"session_id": session_id})).mappings().first()

        if row is None:
            return None

        return QASession(
            session_id=row["session_id"],
            collection_name=row["collection_name"],
            user_id=row["user_id"],
            metadata=row["metadata_json"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def save_message(self, message: QAMessage) -> None:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            await asyncio.to_thread(self.local_repository.save_message, message)
            return

        payload = message.model_dump(mode="json")
        statement = text(
            """
            INSERT INTO qa_messages (
                message_id,
                session_id,
                role,
                query_type,
                question,
                answer,
                decision,
                confidence,
                citations_json,
                evidence_json,
                metadata_json,
                retrieval_trace_id,
                created_at
            ) VALUES (
                :message_id,
                :session_id,
                :role,
                :query_type,
                :question,
                :answer,
                :decision,
                :confidence,
                CAST(:citations_json AS JSONB),
                CAST(:evidence_json AS JSONB),
                CAST(:metadata_json AS JSONB),
                :retrieval_trace_id,
                :created_at
            )
            ON CONFLICT (message_id) DO UPDATE SET
                session_id=EXCLUDED.session_id,
                role=EXCLUDED.role,
                query_type=EXCLUDED.query_type,
                question=EXCLUDED.question,
                answer=EXCLUDED.answer,
                decision=EXCLUDED.decision,
                confidence=EXCLUDED.confidence,
                citations_json=EXCLUDED.citations_json,
                evidence_json=EXCLUDED.evidence_json,
                metadata_json=EXCLUDED.metadata_json,
                retrieval_trace_id=EXCLUDED.retrieval_trace_id,
                created_at=EXCLUDED.created_at
            """
        )

        async with get_async_session(backend="pgvector", database_url=self.database_url) as db_session:
            await db_session.execute(
                statement,
                {
                    "message_id": payload["message_id"],
                    "session_id": payload["session_id"],
                    "role": payload["role"],
                    "query_type": payload.get("query_type", ""),
                    "question": payload.get("question", ""),
                    "answer": payload.get("answer", ""),
                    "decision": payload.get("decision") or "",
                    "confidence": float(payload.get("confidence", 0.0)),
                    "citations_json": self._to_json(payload.get("citations", [])),
                    "evidence_json": self._to_json(payload.get("evidence", [])),
                    "metadata_json": self._to_json(payload.get("metadata", {})),
                    "retrieval_trace_id": payload.get("retrieval_trace_id", ""),
                    "created_at": payload.get("created_at") or self._now_utc(),
                },
            )
            await db_session.commit()

    async def list_messages(self, session_id: str, limit: int = 1000) -> list[QAMessage]:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            return await asyncio.to_thread(self.local_repository.list_messages, session_id, limit)

        query = text(
            """
            SELECT
                message_id,
                session_id,
                role,
                query_type,
                question,
                answer,
                decision,
                confidence,
                citations_json,
                evidence_json,
                metadata_json,
                retrieval_trace_id,
                created_at
            FROM qa_messages
            WHERE session_id = :session_id
            ORDER BY created_at ASC
            LIMIT :limit
            """
        )

        async with get_async_session(backend="pgvector", database_url=self.database_url) as db_session:
            rows = (
                await db_session.execute(
                query,
                {"session_id": session_id, "limit": max(1, int(limit))},
                )
            ).mappings().all()

        return [
            QAMessage(
                message_id=row["message_id"],
                session_id=row["session_id"],
                role=row["role"],
                query_type=row["query_type"],
                question=row["question"],
                answer=row["answer"],
                decision=row["decision"] or None,
                confidence=float(row["confidence"] or 0.0),
                citations=row["citations_json"] or [],
                evidence=row["evidence_json"] or [],
                metadata=row["metadata_json"] or {},
                retrieval_trace_id=row["retrieval_trace_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
