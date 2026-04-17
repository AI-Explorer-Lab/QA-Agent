"""Local development repository backed by sqlite for no-PostgreSQL environments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from database import get_local_dev_database_url, init_local_dev_schema
from domain import Chunk, Document, QAMessage, QASession, RetrievalCandidate, RetrievalTrace


class LocalDevRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = (database_url or get_local_dev_database_url()).strip()
        self.sqlite_path = self._sqlite_path_from_url(self.database_url)
        init_local_dev_schema(self.database_url)

    @staticmethod
    def _sqlite_path_from_url(database_url: str) -> Path:
        if database_url.startswith("sqlite:///"):
            sqlite_path = database_url.replace("sqlite:///", "", 1)
        else:
            sqlite_path = database_url

        resolved = Path(sqlite_path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved.resolve()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=True, default=str)

    @staticmethod
    def _from_json(raw: str, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_document(self, document: Document) -> None:
        self.upsert_documents([document])

    def upsert_documents(self, documents: Iterable[Document]) -> int:
        records = list(documents)
        if not records:
            return 0

        now_iso = self._now_iso()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            for document in records:
                payload = document.model_dump(mode="json")
                cursor.execute(
                    """
                    INSERT INTO pdf_documents (
                        doc_id,
                        collection_name,
                        doc_source,
                        title,
                        doc_hash,
                        page_count,
                        metadata_json,
                        indexed_at,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        collection_name=excluded.collection_name,
                        doc_source=excluded.doc_source,
                        title=excluded.title,
                        doc_hash=excluded.doc_hash,
                        page_count=excluded.page_count,
                        metadata_json=excluded.metadata_json,
                        indexed_at=excluded.indexed_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        payload["doc_id"],
                        payload["collection_name"],
                        payload["doc_source"],
                        payload.get("title", ""),
                        payload.get("doc_hash", ""),
                        int(payload.get("page_count", 0)),
                        self._to_json(payload.get("metadata", {})),
                        str(payload.get("indexed_at") or now_iso),
                        str(payload.get("created_at") or now_iso),
                        str(payload.get("updated_at") or now_iso),
                    ),
                )
            connection.commit()
            return len(records)
        finally:
            connection.close()

    def list_documents_by_collection(self, collection_name: str, limit: int = 1000) -> list[Document]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT * FROM pdf_documents
                WHERE collection_name = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (collection_name, max(1, int(limit))),
            )
            rows = cursor.fetchall()
        finally:
            connection.close()

        documents: list[Document] = []
        for row in rows:
            documents.append(
                Document(
                    doc_id=row["doc_id"],
                    collection_name=row["collection_name"],
                    doc_source=row["doc_source"],
                    title=row["title"],
                    doc_hash=row["doc_hash"],
                    page_count=int(row["page_count"] or 0),
                    metadata=self._from_json(row["metadata_json"], {}),
                    indexed_at=row["indexed_at"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return documents

    def upsert_chunk(self, chunk: Chunk) -> None:
        self.upsert_chunks([chunk])

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        records = list(chunks)
        if not records:
            return 0

        now_iso = self._now_iso()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            for chunk in records:
                payload = chunk.model_dump(mode="json")
                search_text = str(payload.get("search_text") or payload.get("content", ""))
                cursor.execute(
                    """
                    INSERT INTO pdf_chunks (
                        chunk_id,
                        doc_id,
                        collection_name,
                        doc_source,
                        page_idx,
                        page_range,
                        chunk_type,
                        chunk_index,
                        heading_path,
                        level1_title,
                        level2_title,
                        level3_title,
                        table_id,
                        sub_table_id,
                        table_header_text,
                        table_context_text,
                        search_text,
                        content,
                        metadata_json,
                        embedding_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        doc_id=excluded.doc_id,
                        collection_name=excluded.collection_name,
                        doc_source=excluded.doc_source,
                        page_idx=excluded.page_idx,
                        page_range=excluded.page_range,
                        chunk_type=excluded.chunk_type,
                        chunk_index=excluded.chunk_index,
                        heading_path=excluded.heading_path,
                        level1_title=excluded.level1_title,
                        level2_title=excluded.level2_title,
                        level3_title=excluded.level3_title,
                        table_id=excluded.table_id,
                        sub_table_id=excluded.sub_table_id,
                        table_header_text=excluded.table_header_text,
                        table_context_text=excluded.table_context_text,
                        search_text=excluded.search_text,
                        content=excluded.content,
                        metadata_json=excluded.metadata_json,
                        embedding_json=excluded.embedding_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        payload["chunk_id"],
                        payload["doc_id"],
                        payload["collection_name"],
                        payload["doc_source"],
                        payload.get("page_idx"),
                        payload.get("page_range", ""),
                        payload.get("chunk_type", "text"),
                        int(payload.get("chunk_index", 0)),
                        payload.get("heading_path", ""),
                        payload.get("level1_title", ""),
                        payload.get("level2_title", ""),
                        payload.get("level3_title", ""),
                        payload.get("table_id", ""),
                        payload.get("sub_table_id", ""),
                        payload.get("table_header_text", ""),
                        payload.get("table_context_text", ""),
                        search_text,
                        payload.get("content", ""),
                        self._to_json(payload.get("metadata", {})),
                        self._to_json(payload.get("embedding", [])),
                        str(payload.get("created_at") or now_iso),
                        str(payload.get("updated_at") or now_iso),
                    ),
                )
            connection.commit()
            return len(records)
        finally:
            connection.close()

    def list_chunks_by_collection(
        self,
        collection_name: str,
        doc_id: str | None = None,
        limit: int = 1000,
    ) -> list[Chunk]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            if doc_id:
                cursor.execute(
                    """
                    SELECT * FROM pdf_chunks
                    WHERE collection_name = ? AND doc_id = ?
                    ORDER BY chunk_index ASC
                    LIMIT ?
                    """,
                    (collection_name, doc_id, max(1, int(limit))),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM pdf_chunks
                    WHERE collection_name = ?
                    ORDER BY doc_id ASC, chunk_index ASC
                    LIMIT ?
                    """,
                    (collection_name, max(1, int(limit))),
                )
            rows = cursor.fetchall()
        finally:
            connection.close()

        chunks: list[Chunk] = []
        for row in rows:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    collection_name=row["collection_name"],
                    doc_source=row["doc_source"],
                    page_idx=row["page_idx"],
                    page_range=row["page_range"],
                    chunk_type=row["chunk_type"],
                    chunk_index=int(row["chunk_index"] or 0),
                    heading_path=row["heading_path"],
                    level1_title=row["level1_title"],
                    level2_title=row["level2_title"],
                    level3_title=row["level3_title"],
                    table_id=row["table_id"],
                    sub_table_id=row["sub_table_id"],
                    table_header_text=row["table_header_text"],
                    table_context_text=row["table_context_text"],
                    search_text=row["search_text"],
                    content=row["content"],
                    metadata=self._from_json(row["metadata_json"], {}),
                    embedding=self._from_json(row["embedding_json"], []),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return chunks

    def save_session(self, session: QASession) -> None:
        now_iso = self._now_iso()
        payload = session.model_dump(mode="json")

        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO qa_sessions (
                    session_id,
                    collection_name,
                    user_id,
                    metadata_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    collection_name=excluded.collection_name,
                    user_id=excluded.user_id,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    payload["session_id"],
                    payload["collection_name"],
                    payload.get("user_id", ""),
                    self._to_json(payload.get("metadata", {})),
                    str(payload.get("created_at") or now_iso),
                    str(payload.get("updated_at") or now_iso),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def get_session(self, session_id: str) -> QASession | None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM qa_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return QASession(
            session_id=row["session_id"],
            collection_name=row["collection_name"],
            user_id=row["user_id"],
            metadata=self._from_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_message(self, message: QAMessage) -> None:
        payload = message.model_dump(mode="json")
        now_iso = self._now_iso()

        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    role=excluded.role,
                    query_type=excluded.query_type,
                    question=excluded.question,
                    answer=excluded.answer,
                    decision=excluded.decision,
                    confidence=excluded.confidence,
                    citations_json=excluded.citations_json,
                    evidence_json=excluded.evidence_json,
                    metadata_json=excluded.metadata_json,
                    retrieval_trace_id=excluded.retrieval_trace_id,
                    created_at=excluded.created_at
                """,
                (
                    payload["message_id"],
                    payload["session_id"],
                    payload["role"],
                    payload.get("query_type", ""),
                    payload.get("question", ""),
                    payload.get("answer", ""),
                    payload.get("decision") or "",
                    float(payload.get("confidence", 0.0)),
                    self._to_json(payload.get("citations", [])),
                    self._to_json(payload.get("evidence", [])),
                    self._to_json(payload.get("metadata", {})),
                    payload.get("retrieval_trace_id", ""),
                    str(payload.get("created_at") or now_iso),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def list_messages(self, session_id: str, limit: int = 1000) -> list[QAMessage]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT * FROM qa_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, max(1, int(limit))),
            )
            rows = cursor.fetchall()
        finally:
            connection.close()

        messages: list[QAMessage] = []
        for row in rows:
            messages.append(
                QAMessage(
                    message_id=row["message_id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    query_type=row["query_type"],
                    question=row["question"],
                    answer=row["answer"],
                    decision=row["decision"] or None,
                    confidence=float(row["confidence"] or 0.0),
                    citations=self._from_json(row["citations_json"], []),
                    evidence=self._from_json(row["evidence_json"], []),
                    metadata=self._from_json(row["metadata_json"], {}),
                    retrieval_trace_id=row["retrieval_trace_id"],
                    created_at=row["created_at"],
                )
            )
        return messages

    def save_retrieval_trace(self, trace: RetrievalTrace) -> None:
        payload = trace.model_dump(mode="json")

        selected_candidates = [
            candidate.model_dump(mode="json") if isinstance(candidate, RetrievalCandidate) else candidate
            for candidate in trace.selected_candidates
        ]

        retrieval_trace_payload = {
            "dense_candidates": payload.get("dense_candidates", []),
            "sparse_candidates": payload.get("sparse_candidates", []),
            "merged_candidates": payload.get("merged_candidates", []),
            "latency_ms": payload.get("latency_ms", 0.0),
        }

        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    message_id=excluded.message_id,
                    collection_name=excluded.collection_name,
                    question=excluded.question,
                    expanded_queries_json=excluded.expanded_queries_json,
                    retrieval_trace_json=excluded.retrieval_trace_json,
                    rerank_trace_json=excluded.rerank_trace_json,
                    selected_candidates_json=excluded.selected_candidates_json,
                    created_at=excluded.created_at
                """,
                (
                    payload["trace_id"],
                    payload.get("session_id", ""),
                    payload.get("message_id", ""),
                    payload.get("collection_name", ""),
                    payload.get("question", ""),
                    self._to_json(payload.get("expanded_queries", [])),
                    self._to_json(retrieval_trace_payload),
                    self._to_json(payload.get("rerank_trace", {})),
                    self._to_json(selected_candidates),
                    str(payload.get("created_at") or self._now_iso()),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def list_retrieval_traces(self, session_id: str, limit: int = 100) -> list[RetrievalTrace]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT * FROM retrieval_traces
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, max(1, int(limit))),
            )
            rows = cursor.fetchall()
        finally:
            connection.close()

        traces: list[RetrievalTrace] = []
        for row in rows:
            retrieval_payload = self._from_json(row["retrieval_trace_json"], {})
            trace_payload = {
                "trace_id": row["trace_id"],
                "session_id": row["session_id"],
                "message_id": row["message_id"],
                "collection_name": row["collection_name"],
                "question": row["question"],
                "expanded_queries": self._from_json(row["expanded_queries_json"], []),
                "dense_candidates": retrieval_payload.get("dense_candidates", []),
                "sparse_candidates": retrieval_payload.get("sparse_candidates", []),
                "merged_candidates": retrieval_payload.get("merged_candidates", []),
                "selected_candidates": self._from_json(row["selected_candidates_json"], []),
                "rerank_trace": self._from_json(row["rerank_trace_json"], {}),
                "latency_ms": float(retrieval_payload.get("latency_ms", 0.0) or 0.0),
                "created_at": row["created_at"],
            }
            traces.append(RetrievalTrace.model_validate(trace_payload))
        return traces
