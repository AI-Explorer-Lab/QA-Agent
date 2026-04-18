from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import text

from database.connection import get_sqlalchemy_engine


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
        self.engine = get_sqlalchemy_engine(backend="pgvector")

    def load_session(self, session_id: str | None = None, collection_name: str = "default") -> Dict[str, Any]:
        sid = _clean_str(session_id) or str(uuid.uuid4())
        cname = _clean_str(collection_name) or "default"
        self._ensure_session(sid, cname)
        session = self.get_session(sid)
        if session is None:
            return {
                "session_id": sid,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "messages": [],
                "retrieval_traces": [],
            }
        return session

    def save_session(
        self,
        session_id: str,
        user_question: str,
        assistant_payload: Dict[str, Any],
        retrieval_trace: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        sid = _clean_str(session_id) or str(uuid.uuid4())
        trace = dict(retrieval_trace or {})
        cname = _clean_str(trace.get("collection_name") or assistant_payload.get("collection_name")) or "default"
        self._ensure_session(sid, cname)

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        trace_id = _clean_str(trace.get("trace_id")) or str(uuid.uuid4())
        evaluation = dict(trace.get("evaluation") or assistant_payload.get("evaluation") or {})
        timestamp = _utc_now_iso()

        with self.engine.begin() as connection:
            connection.execute(
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
            connection.execute(
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
                        }
                    ),
                    "retrieval_trace_id": trace_id,
                },
            )
            if trace:
                connection.execute(
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
                connection.execute(
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
            connection.execute(
                text("UPDATE qa_sessions SET collection_name=:collection_name, updated_at=NOW() WHERE session_id=:session_id"),
                {"collection_name": cname, "session_id": sid},
            )
        return self.get_session(sid) or self.load_session(sid, cname)

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        sid = _clean_str(session_id)
        if not sid:
            return None
        with self.engine.begin() as connection:
            session = connection.execute(
                text(
                    """
                    SELECT session_id, collection_name, metadata_json, created_at, updated_at
                    FROM qa_sessions
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": sid},
            ).mappings().first()
            if session is None:
                return None

            message_rows = connection.execute(
                text(
                    """
                    SELECT message_id, role, query_type, question, answer, decision, confidence,
                           citations_json, evidence_json, metadata_json, retrieval_trace_id, created_at
                    FROM qa_messages
                    WHERE session_id = :session_id
                    ORDER BY id ASC
                    """
                ),
                {"session_id": sid},
            ).mappings().all()
            trace_rows = connection.execute(
                text(
                    """
                    SELECT trace_id, message_id, collection_name, question, expanded_queries_json,
                           retrieval_trace_json, rerank_trace_json, selected_candidates_json, created_at
                    FROM retrieval_traces
                    WHERE session_id = :session_id
                    ORDER BY id ASC
                    """
                ),
                {"session_id": sid},
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
            "created_at": session.get("created_at").isoformat() if session.get("created_at") else "",
            "updated_at": session.get("updated_at").isoformat() if session.get("updated_at") else "",
            "messages": messages,
            "retrieval_traces": traces,
        }

    def upsert_collection_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        cname = _clean_str(collection_name) or "default"
        with self.engine.begin() as connection:
            count = connection.execute(
                text("SELECT COUNT(*) FROM pdf_chunks WHERE collection_name = :collection_name"),
                {"collection_name": cname},
            ).scalar()
        return {"collection_name": cname, "chunk_count": int(count or 0)}

    def get_collection_chunks(self, collection_name: str) -> List[Dict[str, Any]]:
        cname = _clean_str(collection_name) or "default"
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT chunk_id, doc_id, collection_name, doc_source, page_idx, page_range,
                           chunk_type, chunk_index, heading_path, content AS raw_doc, metadata_json
                    FROM pdf_chunks
                    WHERE collection_name = :collection_name
                    ORDER BY id ASC
                    """
                ),
                {"collection_name": cname},
            ).mappings().all()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata_json"] = _json_loads(item.get("metadata_json"), {})
            result.append(item)
        return result

    def _ensure_session(self, session_id: str, collection_name: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
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


_SESSION_SERVICE = SessionService()


def get_session_service() -> SessionService:
    return _SESSION_SERVICE
