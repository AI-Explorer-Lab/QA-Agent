from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def load_react_guardrails_config() -> Dict[str, Any]:
    return {
        "clarify_enabled": _safe_bool(os.getenv("REACT_CLARIFY_ENABLED"), True),
        "clarify_rag_only": _safe_bool(os.getenv("REACT_CLARIFY_RAG_ONLY"), True),
        "clarify_max_turns": max(0, _safe_int(os.getenv("REACT_CLARIFY_MAX_TURNS"), 1)),
        "evidence_min_docs": max(1, _safe_int(os.getenv("REACT_EVIDENCE_MIN_DOCS"), 2)),
        "evidence_min_top_similarity": max(
            0.0, min(1.0, _safe_float(os.getenv("REACT_EVIDENCE_MIN_TOP_SIMILARITY"), 0.45))
        ),
        "evidence_min_avg_similarity": max(
            0.0, min(1.0, _safe_float(os.getenv("REACT_EVIDENCE_MIN_AVG_SIMILARITY"), 0.30))
        ),
        "evidence_min_overall_score": max(
            0.0, min(1.0, _safe_float(os.getenv("REACT_EVIDENCE_MIN_OVERALL_SCORE"), 0.30))
        ),
        "evidence_retry_limit": max(
            0, _safe_int(os.getenv("REACT_EVIDENCE_RETRY_LIMIT"), 2)
        ),
        "refuse_on_low_evidence": _safe_bool(
            os.getenv("REACT_REFUSE_ON_LOW_EVIDENCE"), True
        ),
    }


def _is_rag_like_query(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    keywords = (
        "财报",
        "年报",
        "季报",
        "文档",
        "资料",
        "报告",
        "pdf",
        "table",
        "表格",
        "chunk",
        "检索",
        "rag",
        "主营业务",
        "分行业",
        "分产品",
        "分地区",
        "营收",
        "净利润",
    )
    return any(keyword in text for keyword in keywords)


def _is_financial_intent(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    financial_keywords = (
        "财报",
        "年报",
        "季报",
        "主营业务",
        "分行业",
        "分产品",
        "分地区",
        "营收",
        "净利润",
        "毛利率",
    )
    return any(keyword in text for keyword in financial_keywords)


def _has_year(query: str) -> bool:
    return bool(re.search(r"(19|20)\d{2}", query or ""))


def _has_company_name(query: str) -> bool:
    text = query or ""
    if re.search(r"[\u4e00-\u9fff]{2,}(公司|集团|股份|科技|银行|证券|基金)", text):
        return True
    if re.search(r"\b[A-Z][A-Za-z0-9&\-. ]{1,30}(Inc|Corp|Ltd|Group)\b", text):
        return True
    return False


def _has_dimension(query: str) -> bool:
    text = query or ""
    dimensions = ("分行业", "分产品", "分地区", "按行业", "按产品", "按地区", "维度")
    return any(d in text for d in dimensions)


def should_clarify_query(
    query: str,
    *,
    rag_only: bool = True,
) -> Tuple[bool, str, List[str]]:
    if not query or not query.strip():
        return False, "", []

    if rag_only and not _is_rag_like_query(query):
        return False, "", []

    if not _is_financial_intent(query):
        return False, "", []

    missing_slots: List[str] = []
    if not _has_company_name(query):
        missing_slots.append("company")
    if not _has_year(query):
        missing_slots.append("year")
    if not _has_dimension(query):
        missing_slots.append("dimension")

    # Only clarify when at least company/year are missing.
    must_missing = {"company", "year"}
    if must_missing.intersection(set(missing_slots)):
        return True, "missing_required_slots_for_rag", missing_slots
    return False, "", []


def build_clarify_question(query: str, missing_slots: List[str]) -> str:
    prompts: List[str] = []
    if "company" in missing_slots:
        prompts.append("公司名称")
    if "year" in missing_slots:
        prompts.append("年份")
    if "dimension" in missing_slots:
        prompts.append("维度（如分行业/分产品/分地区）")

    slot_text = "、".join(prompts) if prompts else "关键信息"
    return (
        f"为了保证检索结果准确，我需要先确认{slot_text}。"
        "请补充后我再基于文档给你结构化回答。"
    )


def summarize_evidence(
    retrieved_docs: List[Dict[str, Any]],
    ragas_evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    docs = retrieved_docs or []
    similarities = [
        max(0.0, min(1.0, _safe_float(doc.get("similarity"), 0.0))) for doc in docs
    ]
    doc_count = len(docs)
    top_similarity = max(similarities) if similarities else 0.0
    avg_similarity = (sum(similarities) / len(similarities)) if similarities else 0.0
    overall_score = ragas_evaluation.get("overall_score")
    if overall_score is not None:
        overall_score = max(0.0, min(1.0, _safe_float(overall_score, 0.0)))
    has_structured_hits = any(
        bool(doc.get("heading_path") or doc.get("table_header_text")) for doc in docs
    )

    return {
        "doc_count": doc_count,
        "top_similarity": round(top_similarity, 4),
        "avg_similarity": round(avg_similarity, 4),
        "overall_score": round(overall_score, 4) if overall_score is not None else None,
        "has_structured_hits": has_structured_hits,
    }


def _evidence_fail_reasons(
    evidence_summary: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    doc_count = int(evidence_summary.get("doc_count", 0))
    if doc_count < int(config["evidence_min_docs"]):
        reasons.append("low_doc_count")

    top_similarity = _safe_float(evidence_summary.get("top_similarity", 0.0), 0.0)
    if top_similarity < float(config["evidence_min_top_similarity"]):
        reasons.append("low_top_similarity")

    avg_similarity = _safe_float(evidence_summary.get("avg_similarity", 0.0), 0.0)
    if avg_similarity < float(config["evidence_min_avg_similarity"]):
        reasons.append("low_avg_similarity")

    overall_score = evidence_summary.get("overall_score")
    if overall_score is not None:
        overall = _safe_float(overall_score, 0.0)
        if overall < float(config["evidence_min_overall_score"]):
            reasons.append("low_overall_score")

    return reasons


def decide_evidence_action(
    evidence_summary: Dict[str, Any],
    retry_count: int,
    config: Dict[str, Any],
) -> Tuple[str, str]:
    reasons = _evidence_fail_reasons(evidence_summary, config)
    if not reasons:
        return "final_answer", "evidence_passed"

    retry_limit = int(config["evidence_retry_limit"])
    if retry_count < retry_limit:
        return "agent_retry", f"evidence_insufficient:{','.join(reasons)}"

    if bool(config["refuse_on_low_evidence"]):
        return "refuse_answer", f"evidence_insufficient:{','.join(reasons)}"

    return "final_answer", f"low_evidence_but_continue:{','.join(reasons)}"


def build_retry_system_prompt(
    question: str,
    evidence_summary: Dict[str, Any],
    reason: str,
) -> str:
    return (
        "Previous retrieval evidence is insufficient for a reliable final answer. "
        "You must call llm_rag again with a refined query that adds missing constraints "
        "or more explicit keywords. "
        f"Original question: {question}\n"
        f"Evidence summary: {evidence_summary}\n"
        f"Reason: {reason}\n"
        "Do not provide final answer now. Perform one more retrieval step first."
    )


def build_refusal_answer(
    question: str,
    evidence_summary: Dict[str, Any],
) -> str:
    return (
        "基于当前检索结果，我暂时无法给出可靠结论。"
        "建议你补充更具体的公司、年份、指标口径或文档范围后再查询。"
        f"（问题：{question}；证据概况：{evidence_summary}）"
    )
