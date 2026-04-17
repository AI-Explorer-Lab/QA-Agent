from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_traces: List[Dict[str, Any]] = field(default_factory=list)


class SessionService:
    """
    Lightweight in-memory session store used by local_dev and tests.

    This intentionally avoids external services so acceptance scripts can run
    without PostgreSQL/Redis while keeping a stable contract for future
    persistence integration.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionRecord] = {}
        self._collections: Dict[str, List[Dict[str, Any]]] = {}

    def load_session(self, session_id: str | None = None) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = SessionRecord(session_id=sid)
            session = self._sessions[sid]
            session.updated_at = _utc_now_iso()
            return self._snapshot_session(session)

    def save_session(
        self,
        session_id: str,
        user_question: str,
        assistant_payload: Dict[str, Any],
        retrieval_trace: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRecord(session_id=session_id)

            session = self._sessions[session_id]
            timestamp = _utc_now_iso()
            user_msg = {
                "message_id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "role": "user",
                "content": user_question,
                "metadata": {},
            }
            assistant_msg = {
                "message_id": str(uuid.uuid4()),
                "timestamp": _utc_now_iso(),
                "role": "assistant",
                "content": assistant_payload.get("answer", ""),
                "metadata": {
                    "decision": assistant_payload.get("decision"),
                    "query_type": assistant_payload.get("query_type"),
                    "confidence": assistant_payload.get("confidence"),
                    "citations": assistant_payload.get("citations", []),
                    "evaluation": assistant_payload.get("evaluation", {}),
                    "skill_trace": assistant_payload.get("skill_trace", {}),
                },
            }
            session.messages.extend([user_msg, assistant_msg])
            if retrieval_trace:
                session.retrieval_traces.append(deepcopy(retrieval_trace))
            session.updated_at = _utc_now_iso()
            return self._snapshot_session(session)

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return self._snapshot_session(session)

    def upsert_collection_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        cname = str(collection_name or "default").strip() or "default"
        with self._lock:
            if force_rebuild or cname not in self._collections:
                self._collections[cname] = []

            existing = {item.get("chunk_id"): item for item in self._collections[cname]}
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                existing[chunk_id] = deepcopy(chunk)

            self._collections[cname] = list(existing.values())
            return {
                "collection_name": cname,
                "chunk_count": len(self._collections[cname]),
            }

    def get_collection_chunks(self, collection_name: str) -> List[Dict[str, Any]]:
        cname = str(collection_name or "default").strip() or "default"
        with self._lock:
            return deepcopy(self._collections.get(cname, []))

    def _snapshot_session(self, session: SessionRecord) -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": deepcopy(session.messages),
            "retrieval_traces": deepcopy(session.retrieval_traces),
        }


_SESSION_SERVICE = SessionService()


def get_session_service() -> SessionService:
    return _SESSION_SERVICE
