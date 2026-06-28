from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import text

from database.connection import get_async_session


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


class SessionService:
    """PostgreSQL-backed session, message, retrieval trace, and evaluation store."""

    def __init__(self) -> None:
        self.backend = "pgvector"

    async def load_session(self, session_id: str | None = None, collection_name: str = "default") -> Dict[str, Any]:
        sid = _clean_str(session_id) or str(uuid.uuid4())
        cname = _clean_str(collection_name) or "default"
        await self._ensure_session(sid, cname)
        session = await self.get_session(sid)
        if session is None:
            return {
                "session_id": sid,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "messages": [],
                "retrieval_traces": [],
            }
        return session

    async def save_session(
        self,
        session_id: str,
        user_question: str,
        assistant_payload: Dict[str, Any],
        retrieval_trace: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        sid = _clean_str(session_id) or str(uuid.uuid4())
        trace = dict(retrieval_trace or {})
        cname = _clean_str(trace.get("collection_name") or assistant_payload.get("collection_name")) or "default"
        await self._ensure_session(sid, cname)

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        trace_id = _clean_str(trace.get("trace_id")) or str(uuid.uuid4())
        evaluation = dict(trace.get("evaluation") or assistant_payload.get("evaluation") or {})
        timestamp = _utc_now_iso()

        async with get_async_session(backend=self.backend) as session:
            await session.execute(
                text(
                    """
                    INSERT INTO qa_messages (
                        message_id, session_id, role, query_type, question, answer, decision,
                        confidence, citations_json, evidence_json, metadata_json,
                        retrieval_trace_id, created_at
                    ) VALUES (
                        :message_id, :session_id, 'user', '', :question, '', '',
                        0, '[]'::jsonb, '[]'::jsonb, '{}'::jsonb, '', NOW()
                    )
                    ON CONFLICT (message_id) DO NOTHING
                    """
                ),
                {
                    "message_id": user_message_id,
                    "session_id": sid,
                    "question": user_question,
                },
            )
            await session.execute(
                text(
                    """
                    INSERT INTO qa_messages (
                        message_id, session_id, role, query_type, question, answer, decision,
                        confidence, citations_json, evidence_json, metadata_json,
                        retrieval_trace_id, created_at
                    ) VALUES (
                        :message_id, :session_id, 'assistant', :query_type, :question, :answer,
                        :decision, :confidence, CAST(:citations_json AS jsonb),
                        CAST(:evidence_json AS jsonb), CAST(:metadata_json AS jsonb),
                        :retrieval_trace_id, NOW()
                    )
                    ON CONFLICT (message_id) DO NOTHING
                    """
                ),
                {
                    "message_id": assistant_message_id,
                    "session_id": sid,
                    "query_type": _clean_str(assistant_payload.get("query_type")),
                    "question": user_question,
                    "answer": _clean_str(assistant_payload.get("answer")),
                    "decision": _clean_str(assistant_payload.get("decision")),
                    "confidence": float(assistant_payload.get("confidence") or 0.0),
                    "citations_json": _json_dumps(assistant_payload.get("citations") or []),
                    "evidence_json": _json_dumps(assistant_payload.get("evidence") or []),
                    "metadata_json": _json_dumps(
                        {
                            "timestamp": timestamp,
                            "decision": assistant_payload.get("decision"),
                            "query_type": assistant_payload.get("query_type"),
                            "confidence": assistant_payload.get("confidence"),
                            "citations": assistant_payload.get("citations") or [],
                            "evaluation": evaluation,
                            "skill_trace": assistant_payload.get("skill_trace") or {},
                            "original_question": assistant_payload.get("original_question") or user_question,
                            "effective_question": assistant_payload.get("effective_question") or user_question,
                            "turn_type": assistant_payload.get("turn_type") or "",
                            "turn_routing": (assistant_payload.get("retrieval_trace") or {}).get("turn_routing") or {},
                        }
                    ),
                    "retrieval_trace_id": trace_id,
                },
            )
            if trace:
                await session.execute(
                    text(
                        """
                        INSERT INTO retrieval_traces (
                            trace_id, session_id, message_id, collection_name, question,
                            expanded_queries_json, retrieval_trace_json, rerank_trace_json,
                            selected_candidates_json, created_at
                        ) VALUES (
                            :trace_id, :session_id, :message_id, :collection_name, :question,
                            CAST(:expanded_queries_json AS jsonb), CAST(:retrieval_trace_json AS jsonb),
                            CAST(:rerank_trace_json AS jsonb), CAST(:selected_candidates_json AS jsonb), NOW()
                        )
                        ON CONFLICT (trace_id) DO UPDATE SET
                            session_id = EXCLUDED.session_id,
                            message_id = EXCLUDED.message_id,
                            collection_name = EXCLUDED.collection_name,
                            question = EXCLUDED.question,
                            expanded_queries_json = EXCLUDED.expanded_queries_json,
                            retrieval_trace_json = EXCLUDED.retrieval_trace_json,
                            rerank_trace_json = EXCLUDED.rerank_trace_json,
                            selected_candidates_json = EXCLUDED.selected_candidates_json
                        """
                    ),
                    {
                        "trace_id": trace_id,
                        "session_id": sid,
                        "message_id": assistant_message_id,
                        "collection_name": cname,
                        "question": user_question,
                        "expanded_queries_json": _json_dumps(trace.get("expanded_queries") or trace.get("query_variants") or []),
                        "retrieval_trace_json": _json_dumps(trace),
                        "rerank_trace_json": _json_dumps(assistant_payload.get("rerank_trace") or {}),
                        "selected_candidates_json": _json_dumps(assistant_payload.get("evidence") or []),
                    },
                )
            if evaluation:
                await session.execute(
                    text(
                        """
                        INSERT INTO evaluation_records (
                            evaluation_id, session_id, message_id, metrics_json, notes, created_at
                        ) VALUES (
                            :evaluation_id, :session_id, :message_id, CAST(:metrics_json AS jsonb), '', NOW()
                        )
                        ON CONFLICT (evaluation_id) DO NOTHING
                        """
                    ),
                    {
                        "evaluation_id": "eval_" + trace_id,
                        "session_id": sid,
                        "message_id": assistant_message_id,
                        "metrics_json": _json_dumps(evaluation),
                    },
                )
            await session.execute(
                text("UPDATE qa_sessions SET collection_name=:collection_name, updated_at=NOW() WHERE session_id=:session_id"),
                {"collection_name": cname, "session_id": sid},
            )
            await session.commit()
        return await self.get_session(sid) or await self.load_session(sid, cname)

    async def get_session(self, session_id: str) -> Dict[str, Any] | None:
        sid = _clean_str(session_id)
        if not sid:
            return None
        async with get_async_session(backend=self.backend) as db_session:
            session = (
                await db_session.execute(
                text(
                    """
                    SELECT session_id, collection_name, metadata_json, created_at, updated_at
                    FROM qa_sessions
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": sid},
                )
            ).mappings().first()
            if session is None:
                return None

            message_rows = (
                await db_session.execute(
                text(
                    """
                    SELECT message_id, role, query_type, question, answer, decision, confidence,
                           citations_json, evidence_json, metadata_json, retrieval_trace_id, created_at
                    FROM qa_messages
                    WHERE session_id = :session_id
                    ORDER BY created_at ASC, message_id ASC
                    """
                ),
                {"session_id": sid},
                )
            ).mappings().all()
            trace_rows = (
                await db_session.execute(
                text(
                    """
                    SELECT trace_id, message_id, collection_name, question, expanded_queries_json,
                           retrieval_trace_json, rerank_trace_json, selected_candidates_json, created_at
                    FROM retrieval_traces
                    WHERE session_id = :session_id
                    ORDER BY created_at ASC, trace_id ASC
                    """
                ),
                {"session_id": sid},
                )
            ).mappings().all()

        messages: List[Dict[str, Any]] = []
        for row in message_rows:
            metadata = _json_loads(row.get("metadata_json"), {})
            role = row.get("role") or ""
            content = row.get("question") if role == "user" else row.get("answer")
            message_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if role == "assistant":
                message_metadata.setdefault("decision", row.get("decision") or "")
                message_metadata.setdefault("query_type", row.get("query_type") or "")
                message_metadata.setdefault("confidence", float(row.get("confidence") or 0.0))
                message_metadata.setdefault("citations", _json_loads(row.get("citations_json"), []))
                message_metadata.setdefault("evidence", _json_loads(row.get("evidence_json"), []))
                message_metadata.setdefault("retrieval_trace_id", row.get("retrieval_trace_id") or "")
            messages.append(
                {
                    "message_id": row.get("message_id"),
                    "timestamp": row.get("created_at").isoformat() if row.get("created_at") else "",
                    "role": role,
                    "content": content or "",
                    "metadata": message_metadata,
                }
            )

        traces: List[Dict[str, Any]] = []
        for row in trace_rows:
            trace_payload = _json_loads(row.get("retrieval_trace_json"), {})
            if not isinstance(trace_payload, dict):
                trace_payload = {}
            trace_payload.setdefault("trace_id", row.get("trace_id") or "")
            trace_payload.setdefault("message_id", row.get("message_id") or "")
            trace_payload.setdefault("collection_name", row.get("collection_name") or "")
            trace_payload.setdefault("question", row.get("question") or "")
            trace_payload.setdefault("expanded_queries", _json_loads(row.get("expanded_queries_json"), []))
            trace_payload.setdefault("rerank_trace", _json_loads(row.get("rerank_trace_json"), {}))
            trace_payload.setdefault("selected_candidates", _json_loads(row.get("selected_candidates_json"), []))
            trace_payload.setdefault("created_at", row.get("created_at").isoformat() if row.get("created_at") else "")
            traces.append(trace_payload)

        return {
            "session_id": session.get("session_id"),
            "collection_name": session.get("collection_name") or "default",
            "metadata": _json_loads(session.get("metadata_json"), {}),
            "created_at": session.get("created_at").isoformat() if session.get("created_at") else "",
            "updated_at": session.get("updated_at").isoformat() if session.get("updated_at") else "",
            "messages": messages,
            "retrieval_traces": traces,
        }

    async def list_sessions(self, collection_name: str = "default", limit: int = 30) -> Dict[str, Any]:
        cname = _clean_str(collection_name) or "default"
        safe_limit = max(1, min(int(limit or 30), 100))
        async with get_async_session(backend=self.backend) as db_session:
            rows = (
                await db_session.execute(
                text(
                    """
                    WITH latest_user AS (
                        SELECT DISTINCT ON (session_id)
                               session_id,
                               question AS last_user_question,
                               created_at AS last_user_at
                        FROM qa_messages
                        WHERE role = 'user'
                        ORDER BY session_id, created_at DESC, message_id DESC
                    ),
                    latest_assistant AS (
                        SELECT DISTINCT ON (session_id)
                               session_id,
                               decision AS last_decision,
                               query_type AS last_query_type,
                               confidence AS last_confidence,
                               citations_json,
                               evidence_json,
                               metadata_json,
                               retrieval_trace_id,
                               created_at AS last_assistant_at
                        FROM qa_messages
                        WHERE role = 'assistant'
                        ORDER BY session_id, created_at DESC, message_id DESC
                    ),
                    message_counts AS (
                        SELECT
                            session_id,
                            COUNT(*) AS message_count,
                            COUNT(*) FILTER (WHERE role = 'user') AS turn_count
                        FROM qa_messages
                        GROUP BY session_id
                    )
                    SELECT
                        s.session_id,
                        s.collection_name,
                        s.metadata_json,
                        s.created_at,
                        s.updated_at,
                        COALESCE(mc.message_count, 0) AS message_count,
                        COALESCE(mc.turn_count, 0) AS turn_count,
                        lu.last_user_question,
                        lu.last_user_at,
                        la.last_decision,
                        la.last_query_type,
                        la.last_confidence,
                        la.citations_json,
                        la.evidence_json,
                        la.metadata_json AS assistant_metadata_json,
                        la.retrieval_trace_id,
                        la.last_assistant_at
                    FROM qa_sessions s
                    LEFT JOIN latest_user lu ON lu.session_id = s.session_id
                    LEFT JOIN latest_assistant la ON la.session_id = s.session_id
                    LEFT JOIN message_counts mc ON mc.session_id = s.session_id
                    WHERE s.collection_name = :collection_name
                    ORDER BY s.updated_at DESC, s.session_id DESC
                    LIMIT :limit
                    """
                ),
                {"collection_name": cname, "limit": safe_limit},
                )
            ).mappings().all()

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            session_metadata = _json_loads(row.get("metadata_json"), {})
            assistant_metadata = _json_loads(row.get("assistant_metadata_json"), {})
            citations = _json_loads(row.get("citations_json"), [])
            evidence = _json_loads(row.get("evidence_json"), [])
            turn_routing = assistant_metadata.get("turn_routing") if isinstance(assistant_metadata, dict) else {}
            if not isinstance(turn_routing, dict):
                turn_routing = {}
            focus = session_metadata.get("conversation_focus") if isinstance(session_metadata, dict) else None
            if not isinstance(focus, dict):
                focus = None
            title = _clean_str(row.get("last_user_question")) or "新对话"
            turn_type = ""
            if isinstance(assistant_metadata, dict):
                turn_type = _clean_str(turn_routing.get("turn_type") or assistant_metadata.get("turn_type"))
            sessions.append(
                {
                    "session_id": row.get("session_id"),
                    "collection_name": row.get("collection_name") or cname,
                    "title": title[:80],
                    "message_count": int(row.get("message_count") or 0),
                    "turn_count": int(row.get("turn_count") or 0),
                    "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else "",
                    "created_at": row.get("created_at").isoformat() if row.get("created_at") else "",
                    "last_user_question": row.get("last_user_question") or "",
                    "last_decision": row.get("last_decision") or "",
                    "last_query_type": row.get("last_query_type") or "",
                    "last_confidence": float(row.get("last_confidence") or 0.0),
                    "citation_count": len(citations) if isinstance(citations, list) else 0,
                    "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
                    "retrieval_trace_id": row.get("retrieval_trace_id") or "",
                    "turn_type": turn_type,
                    "context_source": turn_routing.get("context_source") or "",
                    "conversation_focus": focus,
                    "last_activity_at": (
                        row.get("last_assistant_at").isoformat()
                        if row.get("last_assistant_at")
                        else row.get("last_user_at").isoformat()
                        if row.get("last_user_at")
                        else row.get("updated_at").isoformat()
                        if row.get("updated_at")
                        else ""
                    ),
                }
            )
        return {"collection_name": cname, "sessions": sessions}

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        sid = _clean_str(session_id)
        if not sid:
            return {"session_id": "", "deleted": False, "deleted_counts": {}}

        async with get_async_session(backend=self.backend) as session:
            evaluation_result = await session.execute(
                text("DELETE FROM evaluation_records WHERE session_id = :session_id"),
                {"session_id": sid},
            )
            trace_result = await session.execute(
                text("DELETE FROM retrieval_traces WHERE session_id = :session_id"),
                {"session_id": sid},
            )
            message_result = await session.execute(
                text("DELETE FROM qa_messages WHERE session_id = :session_id"),
                {"session_id": sid},
            )
            session_result = await session.execute(
                text("DELETE FROM qa_sessions WHERE session_id = :session_id"),
                {"session_id": sid},
            )
            await session.commit()

        counts = {
            "evaluation_records": int(evaluation_result.rowcount or 0),
            "retrieval_traces": int(trace_result.rowcount or 0),
            "qa_messages": int(message_result.rowcount or 0),
            "qa_sessions": int(session_result.rowcount or 0),
        }
        return {
            "session_id": sid,
            "deleted": counts["qa_sessions"] > 0,
            "deleted_counts": counts,
        }

    async def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        sid = _clean_str(session_id)
        if not sid:
            return
        async with get_async_session(backend=self.backend) as session:
            current = (
                await session.execute(
                text("SELECT metadata_json FROM qa_sessions WHERE session_id = :session_id"),
                {"session_id": sid},
                )
            ).scalar()
            payload = _json_loads(current, {})
            if not isinstance(payload, dict):
                payload = {}
            payload.update(dict(metadata or {}))
            await session.execute(
                text(
                    """
                    UPDATE qa_sessions
                    SET metadata_json = CAST(:metadata_json AS jsonb), updated_at = NOW()
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": sid, "metadata_json": _json_dumps(payload)},
            )
            await session.commit()

    async def upsert_collection_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        cname = _clean_str(collection_name) or "default"
        async with get_async_session(backend=self.backend) as session:
            count = (
                await session.execute(
                text("SELECT COUNT(*) FROM pdf_chunks WHERE collection_name = :collection_name"),
                {"collection_name": cname},
                )
            ).scalar()
        return {"collection_name": cname, "chunk_count": int(count or 0)}

    async def get_collection_chunks(self, collection_name: str) -> List[Dict[str, Any]]:
        cname = _clean_str(collection_name) or "default"
        async with get_async_session(backend=self.backend) as session:
            rows = (
                await session.execute(
                text(
                    """
                    SELECT chunk_id, doc_id, collection_name, doc_source, page_idx, page_range,
                           chunk_type, chunk_index, heading_path, content AS raw_doc, metadata_json
                    FROM pdf_chunks
                    WHERE collection_name = :collection_name
                    ORDER BY doc_source ASC, chunk_index ASC, chunk_id ASC
                    """
                ),
                {"collection_name": cname},
                )
            ).mappings().all()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata_json"] = _json_loads(item.get("metadata_json"), {})
            result.append(item)
        return result

    async def delete_collection_doc_source(self, collection_name: str, doc_source: str) -> int:
        cname = _clean_str(collection_name) or "default"
        source = _clean_str(doc_source)
        if not source:
            return 0
        async with get_async_session(backend=self.backend) as session:
            result = await session.execute(
                text(
                    """
                    DELETE FROM pdf_chunks
                    WHERE collection_name = :collection_name AND doc_source = :doc_source
                    """
                ),
                {"collection_name": cname, "doc_source": source},
            )
            await session.commit()
        return int(result.rowcount or 0)

    async def _ensure_session(self, session_id: str, collection_name: str) -> None:
        async with get_async_session(backend=self.backend) as session:
            await session.execute(
                text(
                    """
                    INSERT INTO qa_sessions (session_id, collection_name, metadata_json, created_at, updated_at)
                    VALUES (:session_id, :collection_name, '{}'::jsonb, NOW(), NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        collection_name = COALESCE(NULLIF(EXCLUDED.collection_name, ''), qa_sessions.collection_name),
                        updated_at = NOW()
                    """
                ),
                {"session_id": session_id, "collection_name": collection_name},
            )
            await session.commit()


_SESSION_SERVICE = SessionService()


def get_session_service() -> SessionService:
    return _SESSION_SERVICE
