from __future__ import annotations

import re
from typing import Any, Dict, List

from utils.content_normalizer import normalize_whitespace

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_QUOTED_TEXT_RE = re.compile(r"[\"'“”‘’《](.+?)[\"'“”‘’》]")
_VALUE_QUESTION_RE = re.compile(r"(?:^|[？?，,；;。])\s*([^？?，,；;。]+?)(?:分别)?(?:是|为|有|达到)?多少")
_LEADING_TIME_RE = re.compile(r"^(?:在)?(?:19|20)\d{2}年(?:度|末|全年|上半年|下半年|一季度|二季度|三季度|四季度)?(?:的)?")
_METRIC_WORDS = [
    "收入",
    "利润",
    "毛利",
    "成本",
    "预算",
    "现金流",
    "指标",
    "参数",
    "metric",
    "kpi",
]


def _clean_metric_candidate(value: str) -> str:
    text = normalize_whitespace(value, preserve_newlines=False).strip(" ：:，,。；;？?")
    text = re.sub(r"^(?:请问|请查询|查询|统计|计算|告诉我|其中|包括|包含|以及|并且|和|、)\s*", "", text)
    text = _LEADING_TIME_RE.sub("", text).strip(" 的：:，,。；;？?")
    text = re.sub(r"(?:分别)$", "", text).strip(" 的：:，,。；;？?")
    if len(text) < 2:
        return ""
    return text


def _extract_value_question_metrics(text: str) -> List[str]:
    metrics: List[str] = []
    for match in _VALUE_QUESTION_RE.finditer(text):
        metric = _clean_metric_candidate(match.group(1))
        if metric and metric not in metrics:
            metrics.append(metric)
    return metrics


def extract_slots(question: str, query_type: str) -> Dict[str, Any]:
    text = normalize_whitespace(question, preserve_newlines=False)
    lowered = text.lower()

    years = _YEAR_RE.findall(text)
    quoted = [item.strip() for item in _QUOTED_TEXT_RE.findall(text) if item.strip()]

    metrics = _extract_value_question_metrics(text)
    if not metrics:
        metrics = [word for word in _METRIC_WORDS if word in lowered]

    compare_targets: List[str] = []
    if "和" in text:
        parts = [item.strip() for item in text.split("和") if item.strip()]
        if len(parts) >= 2:
            compare_targets = parts[:2]
    if "vs" in lowered:
        parts = [item.strip() for item in re.split(r"\bvs\b", lowered) if item.strip()]
        if len(parts) >= 2:
            compare_targets = parts[:2]

    scope = text if len(text) >= 8 else ""

    slots: Dict[str, Any] = {
        "years": years,
        "metric": "、".join(metrics) if metrics else "",
        "period": years[0] if years else ("报告期" if query_type == "table_qa" and metrics else ""),
        "target_statement": quoted[0] if quoted else "",
        "compare_targets": compare_targets,
        "scope": scope,
    }

    # citation locate can treat non-empty plain question as target statement fallback
    if query_type == "citation_locate" and not slots["target_statement"] and len(text) >= 10:
        slots["target_statement"] = text

    return slots


def required_slots_for_query_type(query_type: str) -> List[str]:
    mapping = {
        "fact_lookup": [],
        "table_qa": ["metric", "period"],
        "summarization": ["scope"],
        "citation_locate": ["target_statement"],
        "report_generation": ["scope"],
        "multi_doc_compare": ["compare_targets"],
        "ambiguous_query": ["scope"],
    }
    return list(mapping.get(query_type, []))


def find_missing_slots(slots: Dict[str, Any], required_slots: List[str]) -> List[str]:
    missing: List[str] = []
    for slot in required_slots:
        value = slots.get(slot)
        if isinstance(value, list):
            if len(value) == 0:
                missing.append(slot)
            elif slot == "compare_targets" and len(value) < 2:
                missing.append(slot)
            continue
        if not value:
            missing.append(slot)
    return missing


def build_clarify_question(query_type: str, missing_slots: List[str]) -> str:
    slot_map = {
        "metric": "你关注的指标",
        "period": "时间范围（例如 2025 年或 Q1）",
        "target_statement": "要定位的原文句子或主题",
        "compare_targets": "至少两个要对比的文档或对象",
        "scope": "文档范围或主题",
    }
    missing_labels = [slot_map.get(slot, slot) for slot in missing_slots]
    if query_type == "ambiguous_query":
        return "问题信息不够完整，请补充文档范围、时间或指标后我再继续。"
    return "为了给出可信答案，请补充：" + "、".join(missing_labels) + "。"


def run_clarify_gate(question: str, query_type: str, collection_name: str) -> Dict[str, Any]:
    slots = extract_slots(question, query_type)
    required_slots = required_slots_for_query_type(query_type)
    missing_slots = find_missing_slots(slots, required_slots)

    if not str(collection_name or "").strip():
        missing_slots.append("collection_name")

    if missing_slots:
        return {
            "decision": "clarify",
            "missing_slots": missing_slots,
            "clarify_question": build_clarify_question(query_type, missing_slots),
            "slots": slots,
        }

    return {
        "decision": "answer",
        "missing_slots": [],
        "clarify_question": "",
        "slots": slots,
    }
