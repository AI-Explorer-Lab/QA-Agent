from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Type

from service.agent.clarify_gate import extract_slots
from service.agent.query_classifier import (
    _CITATION_KEYWORDS,
    _COMPARE_KEYWORDS,
    _REPORT_KEYWORDS,
    _SUMMARY_KEYWORDS,
    _TABLE_KEYWORDS,
    classify_query_type,
)
from service.agent.schemas import QUERY_TYPE_SET, normalize_query_type
from service.agent.skills import SkillDefinition
from utils.content_normalizer import normalize_whitespace

try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = object  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[override]
        return default

EXTRA_TABLE_HINTS = {"revenue", "profit", "gross margin", "cash flow", "cost", "budget", "kpi", "metric", "data", "value"}

INTENT_KEYWORD_GROUPS: Dict[str, set[str]] = {
    "table_qa": set(_TABLE_KEYWORDS) | EXTRA_TABLE_HINTS,
    "summarization": set(_SUMMARY_KEYWORDS),
    "citation_locate": set(_CITATION_KEYWORDS),
    "report_generation": set(_REPORT_KEYWORDS),
    "multi_doc_compare": set(_COMPARE_KEYWORDS),
    "fact_lookup": {"fact", "lookup", "definition", "clause"},
    "ambiguous_query": {"ambiguous", "missing context", "unclear"},
}

KEYWORD_GROUP_NAMES = {
    "table_qa": "_TABLE_KEYWORDS",
    "summarization": "_SUMMARY_KEYWORDS",
    "citation_locate": "_CITATION_KEYWORDS",
    "report_generation": "_REPORT_KEYWORDS",
    "multi_doc_compare": "_COMPARE_KEYWORDS",
    "fact_lookup": "_FACT_LOOKUP_DEFAULT",
    "ambiguous_query": "_AMBIGUOUS_QUERY_DEFAULT",
}

INTENT_NAMES = {
    "table_qa": "table_metric_lookup",
    "summarization": "summarize_document_scope",
    "citation_locate": "locate_source",
    "report_generation": "generate_report",
    "multi_doc_compare": "compare_documents",
    "fact_lookup": "lookup_fact",
    "ambiguous_query": "clarify_ambiguous_query",
}

SLOT_KEYS = ("years", "metric", "period", "target_statement", "compare_targets", "scope", "table_name", "unit", "focus")
FINAL_AUDIT_DECISIONS = {"answer", "retry", "refuse"}
DECISION_RANK = {"answer": 0, "retry": 1, "refuse": 2}
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class IntentClassificationSchema(BaseModel):
    query_type: str
    matched_keyword_group: str = ""
    intent: str = ""
    reason: str = ""


class SlotFillSchema(BaseModel):
    years: List[str] = Field(default_factory=list)
    metric: str = ""
    period: str = ""
    target_statement: str = ""
    compare_targets: List[str] = Field(default_factory=list)
    scope: str = ""
    table_name: str = ""
    unit: str = ""
    focus: str = ""


class EvidenceAuditSchema(BaseModel):
    semantic_decision: str = "retry"
    missing_aspects: List[str] = Field(default_factory=list)
    evidence_coverage: str = "partial"
    conflict_detected: bool = False
    suggested_retry_query: str = ""
    reason: str = ""


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _safe_complete(llm_service: Any, system_prompt: str, user_prompt: str, max_tokens: int = 400) -> Dict[str, Any] | None:
    if llm_service is None:
        return None
    complete = getattr(llm_service, "complete", None)
    if complete is None:
        return None
    try:
        content = await complete(system_prompt, user_prompt, max_tokens=max_tokens)
    except Exception:
        return None
    return _extract_json_object(content or "")


async def _safe_structured_json(
    llm_service: Any,
    system_prompt: str,
    user_payload: Any,
    schema: Type[Any],
    max_tokens: int = 400,
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

    user_prompt = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
    return await _safe_complete(llm_service, system_prompt, user_prompt, max_tokens=max_tokens)

def _clean_str(value: Any) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=False)


def _clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
    else:
        items = [value]

    cleaned: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _normalize_slots(slots: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = dict(slots or {})
    normalized = {
        "years": _clean_list(payload.get("years")),
        "metric": _clean_str(payload.get("metric")),
        "period": _clean_str(payload.get("period")),
        "target_statement": _clean_str(payload.get("target_statement")),
        "compare_targets": _clean_list(payload.get("compare_targets")),
        "scope": _clean_str(payload.get("scope")),
        "table_name": _clean_str(payload.get("table_name")),
        "unit": _clean_str(payload.get("unit")),
        "focus": _clean_str(payload.get("focus")),
    }
    if not normalized["years"] and normalized["period"]:
        normalized["years"] = YEAR_RE.findall(normalized["period"])
    if not normalized["period"] and normalized["years"]:
        normalized["period"] = normalized["years"][0]
    return normalized


def _merge_slots(base: Mapping[str, Any], incoming: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = _normalize_slots(base)
    candidate = _normalize_slots(incoming)
    for key in SLOT_KEYS:
        value = candidate.get(key)
        if key in {"years", "compare_targets"}:
            items = list(merged.get(key) or [])
            for item in value or []:
                if item and item not in items:
                    items.append(item)
            merged[key] = items
            continue
        if value:
            merged[key] = value
    if not merged["period"] and merged["years"]:
        merged["period"] = merged["years"][0]
    return merged


def _evidence_text(evidence: Sequence[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for row in evidence:
        parts.extend(
            [
                str(row.get("content") or ""),
                str(row.get("raw_doc") or ""),
                str(row.get("doc_source") or ""),
                str(row.get("heading_path") or ""),
            ]
        )
        metadata = row.get("metadata") or {}
        if isinstance(metadata, Mapping):
            parts.extend([str(metadata.get("heading_path") or ""), str(metadata.get("page_range") or "")])
    return normalize_whitespace(" ".join(parts), preserve_newlines=False).lower()


def _coarse_tokens(value: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", str(value or "").lower())


def _is_covered(value: Any, corpus: str) -> bool:
    text = _clean_str(value).lower()
    if not text:
        return True
    if text in corpus:
        return True
    tokens = [token for token in _coarse_tokens(text) if len(token.strip()) > 0]
    if not tokens:
        return False
    overlap = sum(1 for token in tokens if token in corpus)
    return overlap / max(1, len(tokens)) >= 0.5


class IntentUnderstandingAgent:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service

    async def classify(self, question: str) -> Dict[str, Any]:
        llm_result = await self._classify_with_llm(question)
        if llm_result is not None:
            return llm_result
        query_type = self._fallback_query_type(question)
        return self._build_result(query_type, source="rule_fallback", reason="Matched the closest fixed intent group.")

    async def _classify_with_llm(self, question: str) -> Dict[str, Any] | None:
        groups = {key: sorted(values) for key, values in INTENT_KEYWORD_GROUPS.items()}
        parsed = await _safe_structured_json(
            self.llm_service,
            "Classify the user intent into exactly one allowed query_type. Return only JSON.",
            {
                "question": question,
                "allowed_query_types": sorted(QUERY_TYPE_SET),
                "keyword_groups": groups,
                "schema": {
                    "query_type": "one allowed query type",
                    "matched_keyword_group": "keyword group name",
                    "intent": "short snake_case intent",
                    "reason": "brief reason",
                },
            },
            schema=IntentClassificationSchema,
            max_tokens=360,
        )
        if not parsed:
            return None
        query_type = normalize_query_type(parsed.get("query_type"))
        if query_type not in QUERY_TYPE_SET:
            return None
        result = self._build_result(
            query_type,
            source="llm_agent",
            reason=_clean_str(parsed.get("reason")) or "LLM mapped the question to a fixed query type.",
        )
        result["intent"] = _clean_str(parsed.get("intent")) or result["intent"]
        matched_group = _clean_str(parsed.get("matched_keyword_group"))
        if matched_group:
            result["matched_keyword_group"] = matched_group
        return result

    @staticmethod
    def _fallback_query_type(question: str) -> str:
        query_type = classify_query_type(question)
        if query_type != "fact_lookup":
            return query_type
        normalized = _clean_str(question).lower()
        if any(hint in normalized for hint in EXTRA_TABLE_HINTS):
            return "table_qa"
        return query_type

    @staticmethod
    def _build_result(query_type: str, source: str, reason: str) -> Dict[str, Any]:
        normalized = normalize_query_type(query_type)
        return {
            "query_type": normalized,
            "matched_keyword_group": KEYWORD_GROUP_NAMES.get(normalized, "_FACT_LOOKUP_DEFAULT"),
            "intent": INTENT_NAMES.get(normalized, "lookup_fact"),
            "reason": reason,
            "source": source,
        }


class SlotFillingAgent:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service

    async def fill(self, question: str, query_type: str, skill: SkillDefinition | None = None) -> Dict[str, Any]:
        rule_slots = self._rule_slots(question, query_type, skill)
        llm_slots = await self._fill_with_llm(question, query_type, rule_slots, skill)
        merged = _merge_slots(rule_slots, llm_slots)
        if skill is not None:
            merged["__skill_name__"] = skill.skill_name
            merged["__missing_required__"] = skill.get_missing_slots(merged)
        return merged

    def _rule_slots(self, question: str, query_type: str, skill: SkillDefinition | None = None) -> Dict[str, Any]:
        del skill
        slots = _normalize_slots(extract_slots(question, query_type))
        years = YEAR_RE.findall(question or "")
        if years:
            slots["years"] = years
            if not slots["period"] or slots["period"] not in years:
                slots["period"] = years[0]
        lowered = _clean_str(question).lower()
        if not slots["metric"]:
            for hint in sorted(EXTRA_TABLE_HINTS, key=len, reverse=True):
                if hint in lowered:
                    slots["metric"] = hint
                    break
        if not slots["table_name"] and "利润表" in question:
            slots["table_name"] = "利润表"
        if not slots["focus"] and any(token in question for token in ("总结", "概括", "分析", "报告")):
            slots["focus"] = "summary"
        return slots

    async def _fill_with_llm(
        self,
        question: str,
        query_type: str,
        rule_slots: Mapping[str, Any],
        skill: SkillDefinition | None = None,
    ) -> Dict[str, Any] | None:
        skill_payload = skill.package_metadata() if skill is not None else {}
        parsed = await _safe_structured_json(
            self.llm_service,
            "Extract slots from the user question. Return only JSON with the schema required by the selected skill package.",
            {
                "question": question,
                "query_type": query_type,
                "skill_package": skill_payload,
                "fixed_schema": {key: [] if key in {"years", "compare_targets"} else "" for key in SLOT_KEYS},
                "rule_slots": dict(rule_slots),
                "instructions": "Respect the selected skill package slot schema. Do not add new fields. Use empty strings or empty lists when absent.",
            },
            schema=SlotFillSchema,
            max_tokens=500,
        )
        if not parsed:
            return None
        return {key: parsed.get(key) for key in SLOT_KEYS}


class EvidenceAuditAgent:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service

    async def audit(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rerank_trace: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        fallback = self._rule_audit(question, query_type, slots, selected_skill, evidence, rerank_trace)
        llm_audit = await self._audit_with_llm(question, query_type, slots, selected_skill, evidence, rerank_trace)
        if llm_audit is None:
            return fallback
        return self._normalize_audit(llm_audit, fallback=fallback)
    async def decide_from_gate(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rule_gate: Mapping[str, Any] | None,
        rerank_trace: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        gate_payload = dict(rule_gate or {})
        audit = await self.audit(
            question=question,
            query_type=query_type,
            slots=slots,
            selected_skill=selected_skill,
            evidence=evidence,
            rerank_trace=rerank_trace,
        )
        audit_payload = self._normalize_audit(audit)
        gate_decision = _clean_str(gate_payload.get("decision") or "refuse").lower()
        if gate_decision not in DECISION_RANK:
            gate_decision = "refuse"

        result = dict(gate_payload)
        result["rule_gate"] = gate_payload
        result["evidence_audit"] = audit_payload
        result.setdefault("reason", gate_payload.get("reason") or audit_payload.get("reason") or gate_decision)
        if audit_payload.get("missing_aspects"):
            result["missing_aspects"] = _clean_list(audit_payload.get("missing_aspects"))

        if gate_decision == "refuse":
            result["decision"] = gate_decision
            result["reason"] = _clean_str(gate_payload.get("reason") or result.get("reason") or audit_payload.get("reason"))
            result["suggested_retry_query"] = ""
            return result

        final_decision = gate_decision
        audit_decision = audit_payload["semantic_decision"]
        if gate_decision == "answer" and DECISION_RANK[audit_decision] > DECISION_RANK[gate_decision]:
            final_decision = audit_decision

        result["decision"] = final_decision
        if final_decision == "answer":
            result["reason"] = _clean_str(gate_payload.get("reason") or audit_payload.get("reason") or "evidence_passed")
            result["suggested_retry_query"] = ""
            return result

        if final_decision == "refuse":
            result["reason"] = _clean_str(gate_payload.get("reason") or audit_payload.get("reason") or "refuse")
            result["suggested_retry_query"] = ""
            return result

        result["reason"] = _clean_str(gate_payload.get("reason") or audit_payload.get("reason") or "retry")
        result["suggested_retry_query"] = self._build_retry_query_from_gate(
            question=question,
            query_type=query_type,
            slots=slots,
            rule_gate=gate_payload,
            audit=audit_payload,
        )
        return result

    @staticmethod
    def _build_retry_query_from_gate(
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        rule_gate: Mapping[str, Any] | None,
        audit: Mapping[str, Any] | None,
    ) -> str:
        slot_values = _normalize_slots(slots)
        gate_reason = _clean_str((rule_gate or {}).get("reason"))
        audit_payload = EvidenceAuditAgent._normalize_audit(audit or {})
        missing_aspects = [item.lower() for item in _clean_list(audit_payload.get("missing_aspects"))]
        current = _clean_str(audit_payload.get("suggested_retry_query"))

        parts: List[str] = []
        if current and not any(token in current for token in ["missing_", "_after_retry", "low_score", "no_evidence"]):
            parts.append(current)

        if query_type == "table_qa":
            parts.extend([slot_values.get("period") or (slot_values.get("years") or [""])[0], slot_values.get("metric") or "", "table"])
            if "table_evidence" in missing_aspects or "missing_table_evidence" in gate_reason:
                parts.extend(["metric", "value", "unit"])
        elif query_type == "multi_doc_compare":
            parts.extend((slot_values.get("compare_targets") or [])[:3])
            parts.extend([slot_values.get("scope") or "", "compare"])
        elif query_type == "citation_locate":
            parts.extend([slot_values.get("target_statement") or "", "source", "page"])
        elif query_type in {"summarization", "report_generation"}:
            parts.extend([slot_values.get("scope") or "", "summary" if query_type == "summarization" else "report"])
        else:
            parts.append(question)

        if not parts or all(not _clean_str(part) for part in parts):
            parts.append(question)

        deduped: List[str] = []
        for item in parts:
            text = _clean_str(item)
            if text and text not in deduped:
                deduped.append(text)

        return " ".join(deduped) if deduped else question


    async def _audit_with_llm(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rerank_trace: Mapping[str, Any] | None,
    ) -> Dict[str, Any] | None:
        evidence_brief = []
        for index, item in enumerate(list(evidence)[:6], start=1):
            evidence_brief.append(
                {
                    "rank": index,
                    "chunk_type": item.get("chunk_type", ""),
                    "doc_source": item.get("doc_source", ""),
                    "content": _clean_str(item.get("content") or item.get("raw_doc"))[:500],
                    "score": item.get("final_score") or item.get("score"),
                }
            )
        parsed = await _safe_structured_json(
            self.llm_service,
            "Audit whether the retrieved evidence semantically covers the question. Return only JSON.",
            {
                "question": question,
                "query_type": query_type,
                "slots": dict(slots or {}),
                "selected_skill": selected_skill,
                "evidence": evidence_brief,
                "rerank_trace": dict(rerank_trace or {}),
                "schema": {
                    "semantic_decision": "answer | retry | refuse",
                    "missing_aspects": [],
                    "evidence_coverage": "sufficient | partial | poor",
                    "conflict_detected": False,
                    "suggested_retry_query": "",
                    "reason": "",
                },
            },
            schema=EvidenceAuditSchema,
            max_tokens=600,
        )
        return parsed

    def _rule_audit(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rerank_trace: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        del selected_skill, rerank_trace
        rows = [dict(item) for item in evidence if isinstance(item, Mapping)]
        slot_values = _normalize_slots(slots)
        if not rows:
            return self._normalize_audit(
                {
                    "semantic_decision": "retry",
                    "missing_aspects": ["evidence"],
                    "evidence_coverage": "poor",
                    "conflict_detected": False,
                    "suggested_retry_query": question,
                    "reason": "No evidence was available for semantic audit.",
                }
            )

        corpus = _evidence_text(rows)
        missing: List[str] = []

        if query_type == "table_qa":
            for key in ("metric", "period"):
                if slot_values.get(key) and not _is_covered(slot_values[key], corpus):
                    missing.append(key)
            if not any(str(row.get("chunk_type") or "") == "table" for row in rows):
                missing.append("table_evidence")
        elif query_type == "citation_locate":
            target = slot_values.get("target_statement")
            if target and not _is_covered(target, corpus):
                missing.append("target_statement")
        elif query_type == "multi_doc_compare":
            for target in slot_values.get("compare_targets") or []:
                if target and not _is_covered(target, corpus):
                    missing.append(f"compare_target:{target}")
        elif query_type in {"summarization", "report_generation"} and slot_values.get("scope"):
            if not _is_covered(slot_values["scope"], corpus):
                missing.append("scope")

        conflict_detected = self._detect_numeric_conflict(query_type, rows)
        if missing:
            suggested = " ".join([question, str(slot_values.get("metric") or ""), str(slot_values.get("period") or "")]).strip()
            return self._normalize_audit(
                {
                    "semantic_decision": "retry",
                    "missing_aspects": missing,
                    "evidence_coverage": "partial",
                    "conflict_detected": conflict_detected,
                    "suggested_retry_query": suggested or question,
                    "reason": "Evidence does not clearly cover required semantic aspects.",
                }
            )

        return self._normalize_audit(
            {
                "semantic_decision": "answer",
                "missing_aspects": [],
                "evidence_coverage": "sufficient",
                "conflict_detected": conflict_detected,
                "suggested_retry_query": "",
                "reason": "Evidence semantically covers the required aspects.",
            }
        )

    @staticmethod
    def _detect_numeric_conflict(query_type: str, evidence: Sequence[Mapping[str, Any]]) -> bool:
        if query_type != "table_qa":
            return False
        values: List[str] = []
        for row in evidence[:5]:
            text = str(row.get("content") or row.get("raw_doc") or "")
            values.extend(NUMBER_RE.findall(text)[:2])
        return len(set(values)) >= 3

    @staticmethod
    def _normalize_audit(audit: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        base = dict(fallback or {})
        decision = _clean_str(audit.get("semantic_decision") or base.get("semantic_decision") or "answer").lower()
        if decision not in FINAL_AUDIT_DECISIONS:
            decision = "retry"
        coverage = _clean_str(audit.get("evidence_coverage") or base.get("evidence_coverage") or "partial").lower()
        if coverage not in {"sufficient", "partial", "poor"}:
            coverage = "partial"
        return {
            "semantic_decision": decision,
            "missing_aspects": _clean_list(audit.get("missing_aspects") or base.get("missing_aspects")),
            "evidence_coverage": coverage,
            "conflict_detected": bool(audit.get("conflict_detected", base.get("conflict_detected", False))),
            "suggested_retry_query": _clean_str(audit.get("suggested_retry_query") or base.get("suggested_retry_query")),
            "reason": _clean_str(audit.get("reason") or base.get("reason") or "Evidence audit completed."),
            "source": _clean_str(audit.get("source") or base.get("source") or "evidence_audit_agent"),
        }


def merge_audit_and_rule_gate(rule_gate: Mapping[str, Any], audit: Mapping[str, Any]) -> Dict[str, Any]:
    rule = dict(rule_gate or {})
    audit_payload = EvidenceAuditAgent._normalize_audit(audit)
    rule_decision = str(rule.get("decision") or "refuse").strip().lower()
    if rule_decision not in DECISION_RANK:
        rule_decision = "refuse"
    audit_decision = audit_payload["semantic_decision"]
    final_decision = rule_decision
    if DECISION_RANK[audit_decision] > DECISION_RANK[rule_decision]:
        final_decision = audit_decision

    merged = dict(rule)
    merged["decision"] = final_decision
    merged["rule_decision"] = rule_decision
    merged["semantic_decision"] = audit_decision
    merged["evidence_audit"] = audit_payload
    if final_decision != rule_decision:
        merged["reason"] = audit_payload.get("reason") or f"semantic_{audit_decision}"
    else:
        merged.setdefault("reason", rule.get("reason") or audit_payload.get("reason") or final_decision)
    if audit_payload.get("suggested_retry_query"):
        merged["suggested_retry_query"] = audit_payload["suggested_retry_query"]
    if audit_payload.get("missing_aspects"):
        merged.setdefault("missing_aspects", audit_payload["missing_aspects"])
    return merged


