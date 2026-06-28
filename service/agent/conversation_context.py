from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence, Type

from service.agent.clarify_gate import extract_slots
from utils.content_normalizer import normalize_whitespace

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic is available in normal runtime
    BaseModel = object  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[override]
        return default


TURN_TYPES = {"new_rag_query", "follow_up", "clarification_reply", "citation_followup", "meta_question"}
CONTEXT_SOURCES = {"none", "clarification_pending", "conversation_focus", "recent_history", "last_evidence"}
LOW_CONFIDENCE_THRESHOLD = 0.55


class TurnRouteSchema(BaseModel):
    turn_type: str = "new_rag_query"
    should_use_history: bool = False
    context_source: str = "none"
    effective_question: str = ""
    history_refs: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


def _clean_str(value: Any) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=False)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _safe_structured_json(
    llm_service: Any,
    system_prompt: str,
    user_payload: Any,
    schema: Type[Any],
    max_tokens: int = 500,
) -> Dict[str, Any] | None:
    if llm_service is None:
        return None

    structured_json = getattr(llm_service, "structured_json", None)
    if callable(structured_json):
        try:
            payload = await structured_json(
                system_prompt=system_prompt,
                user_payload=user_payload,
                schema=schema,
                max_tokens=max_tokens,
            )
        except Exception:
            payload = None

        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "dict"):
            payload = payload.dict()
        if isinstance(payload, Mapping):
            return dict(payload)

    complete = getattr(llm_service, "complete", None)
    if not callable(complete):
        return None
    try:
        content = await complete(system_prompt, _json_dumps(user_payload), max_tokens=max_tokens)
    except Exception:
        return None
    return _extract_json_object(content or "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _assistant_status(metadata: Mapping[str, Any]) -> str:
    decision = _clean_str(metadata.get("decision")).lower()
    if decision == "clarify":
        return "clarification_pending"
    if decision == "answer":
        return "completed"
    if decision == "refuse":
        return "refused"
    return "unknown"


def _metadata_slots(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    skill_trace = metadata.get("skill_trace") or {}
    if isinstance(skill_trace, Mapping):
        slots = skill_trace.get("slots") or {}
        return dict(slots) if isinstance(slots, Mapping) else {}
    return {}


def _last_completed_focus(turns: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    for turn in reversed(turns):
        if turn.get("status") != "completed":
            continue
        slots = turn.get("slots") or {}
        if not isinstance(slots, Mapping):
            slots = {}
        focus = {
            "active_topic": turn.get("effective_question") or turn.get("user_question") or "",
            "company": slots.get("company") or slots.get("entity") or "",
            "period": slots.get("period") or "",
            "years": slots.get("years") or [],
            "metric": slots.get("metric") or "",
            "scope": slots.get("scope") or "",
            "doc_scope": turn.get("doc_sources") or [],
            "last_query_type": turn.get("query_type") or "",
            "updated_at_turn": turn.get("turn_id") or "",
        }
        return {key: value for key, value in focus.items() if value not in ("", [], None)}
    return None


def _build_turns(messages: Sequence[Mapping[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    pending_user: Mapping[str, Any] | None = None
    turn_index = 0
    for message in messages:
        role = _clean_str(message.get("role")).lower()
        if role == "user":
            pending_user = message
            continue
        if role != "assistant" or pending_user is None:
            continue

        turn_index += 1
        metadata = message.get("metadata") or {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        citations = _safe_list(metadata.get("citations"))
        evidence = _safe_list(metadata.get("evidence"))
        slots = _metadata_slots(metadata)
        doc_sources: List[str] = []
        for item in evidence:
            if isinstance(item, Mapping):
                doc_source = _clean_str(item.get("doc_source"))
                if doc_source and doc_source not in doc_sources:
                    doc_sources.append(doc_source)

        turns.append(
            {
                "turn_id": f"turn_{turn_index}",
                "status": _assistant_status(metadata),
                "user_question": _clean_str(pending_user.get("content")),
                "effective_question": _clean_str(metadata.get("effective_question") or pending_user.get("content")),
                "answer_summary": _clean_str(message.get("content"))[:240],
                "query_type": _clean_str(metadata.get("query_type")),
                "slots": slots,
                "citations_available": bool(citations),
                "evidence_available": bool(evidence),
                "doc_sources": doc_sources,
                "message_id": _clean_str(message.get("message_id")),
                "retrieval_trace_id": _clean_str(metadata.get("retrieval_trace_id")),
            }
        )
        pending_user = None
    return turns[-max(1, int(max_turns)) :]


class ConversationContextService:
    def __init__(
        self,
        llm_service: Any | None = None,
        history_window: int = 5,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.llm_service = llm_service
        self.history_window = max(1, int(history_window))
        self.low_confidence_threshold = float(low_confidence_threshold)

    def build_state(self, session: Mapping[str, Any], current_question: str, collection_name: str) -> Dict[str, Any]:
        messages = _safe_list(session.get("messages"))
        turns = _build_turns([item for item in messages if isinstance(item, Mapping)], self.history_window)
        metadata = session.get("metadata") or {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        conversation_focus = metadata.get("conversation_focus")
        if not isinstance(conversation_focus, Mapping):
            conversation_focus = _last_completed_focus(turns)
        else:
            conversation_focus = dict(conversation_focus)

        latest_clarification_pending = None
        if turns and turns[-1].get("status") == "clarification_pending":
            latest = turns[-1]
            slots = latest.get("slots") or {}
            missing = []
            if isinstance(slots, Mapping):
                missing = _safe_list(slots.get("__missing_required__"))
            latest_clarification_pending = {
                "turn_id": latest.get("turn_id") or "",
                "original_question": latest.get("effective_question") or latest.get("user_question") or "",
                "missing_slots": missing,
                "clarify_question": latest.get("answer_summary") or "",
            }

        last_evidence: List[Any] = []
        last_citations: List[Any] = []
        for message in reversed(messages):
            if not isinstance(message, Mapping) or _clean_str(message.get("role")).lower() != "assistant":
                continue
            metadata = message.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                continue
            last_evidence = _safe_list(metadata.get("evidence"))
            last_citations = _safe_list(metadata.get("citations"))
            break

        return {
            "current_question": _clean_str(current_question),
            "collection_name": _clean_str(collection_name) or _clean_str(session.get("collection_name")) or "default",
            "recent_history": turns,
            "latest_clarification_pending": latest_clarification_pending,
            "conversation_focus": conversation_focus or None,
            "last_evidence": last_evidence,
            "last_citations": last_citations,
        }

    async def route_turn(self, conversation_state: Mapping[str, Any]) -> Dict[str, Any]:
        current_question = _clean_str(conversation_state.get("current_question"))
        fallback = {
            "turn_type": "new_rag_query",
            "should_use_history": False,
            "context_source": "none",
            "effective_question": current_question,
            "history_refs": [],
            "missing_info": [],
            "confidence": 0.0,
            "reason": "LLM turn router unavailable; defaulted to a standalone RAG query.",
        }
        parsed = await _safe_structured_json(
            self.llm_service,
            (
                "You route RAG conversation turns. Return only JSON. "
                "Do not answer the user question. Decide whether history is needed, "
                "rewrite the current turn into a complete effective_question when needed, "
                "and list exactly which history refs are used."
            ),
            {
                "allowed_turn_types": sorted(TURN_TYPES),
                "allowed_context_sources": sorted(CONTEXT_SOURCES),
                "routing_policy": [
                    "Use clarification_pending only when the current question answers the latest clarification request.",
                    "Use conversation_focus for clear follow-up questions that omit subject, period, document, or metric.",
                    "Use last_evidence for citation/source/page follow-ups.",
                    "Use none for a standalone new RAG query.",
                    "If confidence is low or the effective question cannot be completed, include missing_info.",
                ],
                "conversation_state": conversation_state,
                "schema": {
                    "turn_type": "one allowed turn type",
                    "should_use_history": "boolean",
                    "context_source": "one allowed context source",
                    "effective_question": "complete question for downstream RAG",
                    "history_refs": ["turn ids or focus/last_evidence"],
                    "missing_info": ["missing info if clarification is safer"],
                    "confidence": "0 to 1",
                    "reason": "short reason",
                },
            },
            schema=TurnRouteSchema,
            max_tokens=700,
        )
        if not parsed:
            return fallback
        return self._normalize_route(parsed, fallback)

    def apply_policy(self, conversation_state: Mapping[str, Any], route: Mapping[str, Any]) -> Dict[str, Any]:
        current_question = _clean_str(conversation_state.get("current_question"))
        normalized = self._normalize_route(route, {"effective_question": current_question})
        turn_type = normalized["turn_type"]
        context_source = normalized["context_source"]
        confidence = _safe_float(normalized.get("confidence"))

        pending = conversation_state.get("latest_clarification_pending")
        focus = conversation_state.get("conversation_focus")
        last_evidence = _safe_list(conversation_state.get("last_evidence"))
        last_citations = _safe_list(conversation_state.get("last_citations"))

        if confidence and confidence < self.low_confidence_threshold:
            normalized.update(
                {
                    "turn_type": "clarification_reply" if pending else "new_rag_query",
                    "should_use_history": bool(pending),
                    "context_source": "clarification_pending" if pending else "none",
                    "requires_clarification": True,
                    "policy_reason": "turn_router_low_confidence",
                }
            )
            if not normalized.get("missing_info"):
                normalized["missing_info"] = ["conversation_context"]
            return normalized

        if turn_type == "clarification_reply":
            if pending:
                normalized["context_source"] = "clarification_pending"
                normalized["should_use_history"] = True
                if not normalized.get("history_refs"):
                    normalized["history_refs"] = [pending.get("turn_id", "clarification_pending")]
                if not normalized.get("effective_question") or normalized["effective_question"] == current_question:
                    normalized["effective_question"] = _clean_str(f"{pending.get('original_question', '')} {current_question}")
            else:
                normalized.update({"turn_type": "new_rag_query", "context_source": "none", "should_use_history": False})

        elif turn_type == "follow_up":
            if focus:
                normalized["context_source"] = "conversation_focus"
                normalized["should_use_history"] = True
                if not normalized.get("history_refs"):
                    normalized["history_refs"] = ["focus"]
            elif not normalized.get("history_refs"):
                normalized.update({"turn_type": "new_rag_query", "context_source": "none", "should_use_history": False})

        elif turn_type == "citation_followup":
            if last_evidence or last_citations:
                normalized["context_source"] = "last_evidence"
                normalized["should_use_history"] = True
                if not normalized.get("history_refs"):
                    normalized["history_refs"] = ["last_evidence"]
            else:
                normalized.update({"turn_type": "new_rag_query", "context_source": "none", "should_use_history": False})

        elif turn_type == "meta_question":
            normalized["should_use_history"] = True
            if context_source not in CONTEXT_SOURCES or context_source == "none":
                normalized["context_source"] = "recent_history"

        else:
            normalized.update({"turn_type": "new_rag_query", "context_source": "none", "should_use_history": False, "history_refs": []})

        if not normalized.get("effective_question"):
            normalized["effective_question"] = current_question
        normalized["requires_clarification"] = bool(normalized.get("requires_clarification", False))
        return normalized

    async def prepare_context(self, session: Mapping[str, Any], current_question: str, collection_name: str) -> Dict[str, Any]:
        state = self.build_state(session, current_question, collection_name)
        if (
            not state.get("recent_history")
            and not state.get("latest_clarification_pending")
            and not state.get("conversation_focus")
            and not state.get("last_evidence")
            and not state.get("last_citations")
        ):
            question = _clean_str(current_question)
            route = {
                "turn_type": "new_rag_query",
                "should_use_history": False,
                "context_source": "none",
                "effective_question": question,
                "history_refs": [],
                "missing_info": [],
                "confidence": 1.0,
                "reason": "No conversation history; skipped LLM turn routing.",
                "requires_clarification": False,
                "policy_reason": "new_session_fast_path",
            }
            return {
                "conversation_state": state,
                "turn_route": route,
                "original_question": question,
                "effective_question": question,
            }
        route = await self.route_turn(state)
        policy_route = self.apply_policy(state, route)
        return {
            "conversation_state": state,
            "turn_route": policy_route,
            "original_question": _clean_str(current_question),
            "effective_question": _clean_str(policy_route.get("effective_question")) or _clean_str(current_question),
        }

    def build_focus_after_response(
        self,
        previous_focus: Mapping[str, Any] | None,
        effective_question: str,
        query_type: str,
        slots: Mapping[str, Any],
        response: Mapping[str, Any],
        turn_route: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        decision = _clean_str(response.get("decision")).lower()
        if decision == "refuse":
            return dict(previous_focus or {}) or None
        if decision == "clarify":
            return dict(previous_focus or {}) or None

        route_type = _clean_str(turn_route.get("turn_type"))
        focus = {} if route_type == "new_rag_query" else dict(previous_focus or {})
        focus.update(
            {
                "active_topic": _clean_str(effective_question),
                "last_query_type": _clean_str(query_type),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        normalized_slots = extract_slots(effective_question, query_type)
        merged_slots = dict(normalized_slots)
        merged_slots.update({key: value for key, value in dict(slots or {}).items() if value not in ("", [], None)})
        for key in ("company", "entity", "period", "years", "metric", "scope", "table_name", "unit", "focus"):
            value = merged_slots.get(key)
            if value not in ("", [], None):
                focus[key] = value

        doc_scope: List[str] = []
        for item in _safe_list(response.get("evidence")):
            if isinstance(item, Mapping):
                doc_source = _clean_str(item.get("doc_source"))
                if doc_source and doc_source not in doc_scope:
                    doc_scope.append(doc_source)
        if doc_scope:
            focus["doc_scope"] = doc_scope
        return {key: value for key, value in focus.items() if value not in ("", [], None)}

    @staticmethod
    def _normalize_route(route: Mapping[str, Any], fallback: Mapping[str, Any]) -> Dict[str, Any]:
        current_fallback = dict(fallback)
        turn_type = _clean_str(route.get("turn_type") or current_fallback.get("turn_type") or "new_rag_query")
        if turn_type not in TURN_TYPES:
            turn_type = "new_rag_query"
        context_source = _clean_str(route.get("context_source") or current_fallback.get("context_source") or "none")
        if context_source not in CONTEXT_SOURCES:
            context_source = "none"
        effective_question = _clean_str(route.get("effective_question") or current_fallback.get("effective_question"))
        return {
            "turn_type": turn_type,
            "should_use_history": bool(route.get("should_use_history", current_fallback.get("should_use_history", False))),
            "context_source": context_source,
            "effective_question": effective_question,
            "history_refs": [str(item) for item in _safe_list(route.get("history_refs", current_fallback.get("history_refs", []))) if _clean_str(item)],
            "missing_info": [str(item) for item in _safe_list(route.get("missing_info", current_fallback.get("missing_info", []))) if _clean_str(item)],
            "confidence": max(0.0, min(1.0, _safe_float(route.get("confidence", current_fallback.get("confidence", 0.0))))),
            "reason": _clean_str(route.get("reason") or current_fallback.get("reason")),
        }
