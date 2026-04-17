from __future__ import annotations

from typing import Any, Dict, List


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def evaluate_qa_result(
    question: str,
    answer: str,
    decision: str,
    citations: List[Dict[str, Any]] | None = None,
    evidence: List[Dict[str, Any]] | None = None,
) -> Dict[str, float | int | str]:
    citations = citations or []
    evidence = evidence or []

    has_answer = bool((answer or "").strip())
    evidence_count = len(evidence)
    citation_count = len(citations)

    citation_density = _safe_div(citation_count, max(1, evidence_count))
    grounding = 1.0 if decision != "answer" else min(1.0, citation_density)

    if decision == "answer" and not has_answer:
        completeness = 0.0
    elif decision in {"clarify", "refuse"}:
        completeness = 1.0
    else:
        completeness = min(1.0, _safe_div(len(answer), 120.0))

    consistency = 1.0
    if decision == "answer" and evidence_count == 0:
        consistency = 0.0

    overall = (grounding + completeness + consistency) / 3.0
    confidence = min(1.0, max(0.0, overall))

    return {
        "metric": "local_ragas_compatible",
        "question_length": len(question or ""),
        "answer_length": len(answer or ""),
        "evidence_count": evidence_count,
        "citation_count": citation_count,
        "grounding_score": round(grounding, 4),
        "completeness_score": round(completeness, 4),
        "consistency_score": round(consistency, 4),
        "overall_score": round(overall, 4),
        "confidence": round(confidence, 4),
    }
