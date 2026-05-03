from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from service.agent.controlled_agents import EvidenceAuditAgent


def _score(row: Dict[str, Any]) -> float:
    try:
        return float(
            row.get("final_score")
            or row.get("score")
            or row.get("dense_score")
            or row.get("bm25_score")
            or 0.0
        )
    except Exception:
        return 0.0


def _confidence(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    values = [_score(row) for row in rows]
    top = max(values) if values else 0.0
    avg = sum(values) / max(1, len(values))
    return round(max(0.0, min(1.0, (top + avg) / 2.0)), 4)


class EvidenceGate:
    def __init__(
        self,
        evidence_min_docs: int = 1,
        evidence_min_top_score: float = 0.45,
        evidence_min_avg_score: float = 0.30,
        retry_limit: int = 2,
        refuse_on_low_evidence: bool = True,
    ) -> None:
        self.evidence_min_docs = max(1, int(evidence_min_docs))
        self.evidence_min_top_score = float(evidence_min_top_score)
        self.evidence_min_avg_score = float(evidence_min_avg_score)
        self.retry_limit = max(0, int(retry_limit))
        self.refuse_on_low_evidence = bool(refuse_on_low_evidence)

    def evaluate(
        self,
        evidence: List[Dict[str, Any]],
        query_type: str,
        retry_count: int = 0,
        table_evidence_quota: int = 2,
        slots: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        slots = slots or {}
        rows = [dict(item) for item in evidence]
        if not rows:
            decision = "retry" if retry_count < self.retry_limit else "refuse"
            return {
                "decision": decision,
                "reason": "no_evidence" if decision == "retry" else "no_evidence_after_retry",
                "confidence": 0.0,
            }

        scores = [_score(row) for row in rows]
        top_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / max(1, len(scores))
        docs = {
            str(row.get("doc_id") or row.get("doc_source") or "")
            for row in rows
            if row.get("doc_id") or row.get("doc_source")
        }
        table_count = sum(1 for row in rows if str(row.get("chunk_type") or "") == "table")

        if query_type == "table_qa" and table_count < max(1, int(table_evidence_quota)):
            if retry_count < self.retry_limit:
                return {
                    "decision": "retry",
                    "reason": "missing_table_evidence",
                    "confidence": _confidence(rows),
                }
            return {
                "decision": "refuse",
                "reason": "missing_table_evidence_after_retry",
                "confidence": _confidence(rows),
            }

        if query_type == "multi_doc_compare" and len(docs) < 2:
            if retry_count < self.retry_limit:
                return {
                    "decision": "retry",
                    "reason": "multi_doc_evidence_missing",
                    "confidence": _confidence(rows),
                }
            return {
                "decision": "refuse",
                "reason": "multi_doc_evidence_missing_after_retry",
                "confidence": _confidence(rows),
            }

        coverage_sensitive_types = {"summarization", "report_generation", "multi_doc_compare"}
        if query_type in coverage_sensitive_types and len(rows) < self.evidence_min_docs:
            if retry_count < self.retry_limit:
                return {
                    "decision": "retry",
                    "reason": "insufficient_doc_coverage",
                    "confidence": _confidence(rows),
                }
            return {
                "decision": "refuse",
                "reason": "insufficient_doc_coverage_after_retry",
                "confidence": _confidence(rows),
            }

        if top_score < self.evidence_min_top_score or avg_score < self.evidence_min_avg_score:
            if retry_count < self.retry_limit:
                return {
                    "decision": "retry",
                    "reason": "low_score_retry",
                    "confidence": _confidence(rows),
                }
            return {
                "decision": "refuse" if self.refuse_on_low_evidence else "answer",
                "reason": "low_score",
                "confidence": _confidence(rows),
            }

        return {
            "decision": "answer",
            "reason": "evidence_passed",
            "top_score": round(top_score, 4),
            "avg_score": round(avg_score, 4),
            "doc_count": len(docs),
            "table_evidence_count": table_count,
            "confidence": _confidence(rows),
        }


class EvidenceDecisionEngine:
    def __init__(
        self,
        llm_service: Any | None = None,
        evidence_min_docs: int = 1,
        evidence_min_top_score: float = 0.45,
        evidence_min_avg_score: float = 0.30,
        retry_limit: int = 2,
        refuse_on_low_evidence: bool = True,
    ) -> None:
        self.retry_limit = max(0, int(retry_limit))
        self.rule_gate = EvidenceGate(
            evidence_min_docs=evidence_min_docs,
            evidence_min_top_score=evidence_min_top_score,
            evidence_min_avg_score=evidence_min_avg_score,
            retry_limit=retry_limit,
            refuse_on_low_evidence=refuse_on_low_evidence,
        )
        self.evidence_agent = EvidenceAuditAgent(llm_service)

    async def evaluate(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rerank_trace: Mapping[str, Any] | None = None,
        retry_count: int = 0,
        table_evidence_quota: int = 2,
    ) -> Dict[str, Any]:
        rows = [dict(item) for item in evidence if isinstance(item, Mapping)]
        rule_gate = self.rule_gate.evaluate(
            rows,
            query_type=query_type,
            retry_count=retry_count,
            table_evidence_quota=table_evidence_quota,
            slots=dict(slots or {}),
        )
        return await self.evidence_agent.decide_from_gate(
            question=question,
            query_type=query_type,
            slots=slots,
            selected_skill=selected_skill,
            evidence=rows,
            rule_gate=rule_gate,
            rerank_trace=rerank_trace,
        )


def run_evidence_gate(
    query_type: str,
    evidence: List[Dict[str, Any]],
    slots: Dict[str, Any] | None = None,
    retry_count: int = 0,
    retry_limit: int = 2,
    min_top_score: float = 0.45,
    min_avg_score: float = 0.30,
    table_evidence_quota: int = 2,
    refuse_on_low_evidence: bool = True,
) -> Dict[str, Any]:
    gate = EvidenceGate(
        evidence_min_top_score=min_top_score,
        evidence_min_avg_score=min_avg_score,
        retry_limit=retry_limit,
        refuse_on_low_evidence=refuse_on_low_evidence,
    )
    return gate.evaluate(
        evidence=list(evidence or []),
        query_type=query_type,
        retry_count=retry_count,
        table_evidence_quota=table_evidence_quota,
        slots=slots,
    )
