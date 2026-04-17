from __future__ import annotations

import re

from service.agent.schemas import normalize_query_type
from utils.content_normalizer import normalize_whitespace

_TABLE_KEYWORDS = {
    "表",
    "表格",
    "指标",
    "数据",
    "数值",
    "同比",
    "环比",
    "毛利率",
    "收入",
    "成本",
    "利润",
    "参数",
    "table",
    "metric",
}

_SUMMARY_KEYWORDS = {
    "总结",
    "概述",
    "摘要",
    "归纳",
    "overview",
    "summary",
}

_CITATION_KEYWORDS = {
    "出处",
    "原文",
    "哪一页",
    "页码",
    "引用",
    "citation",
    "source",
    "locate",
}

_REPORT_KEYWORDS = {
    "报告",
    "汇报",
    "生成报告",
    "分析报告",
    "report",
}

_COMPARE_KEYWORDS = {
    "对比",
    "比较",
    "差异",
    "区别",
    "versus",
    "vs",
    "compare",
}

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _contains_any(text: str, words: set[str]) -> bool:
    return any(word in text for word in words)


def classify_query_type(question: str) -> str:
    normalized = normalize_whitespace(question, preserve_newlines=False).lower()
    if not normalized:
        return "ambiguous_query"

    # Very short or deictic prompts are usually underspecified.
    if len(normalized) <= 4:
        return "ambiguous_query"

    if _contains_any(normalized, _COMPARE_KEYWORDS):
        return "multi_doc_compare"

    if _contains_any(normalized, _REPORT_KEYWORDS):
        return "report_generation"

    if _contains_any(normalized, _CITATION_KEYWORDS):
        return "citation_locate"

    if _contains_any(normalized, _TABLE_KEYWORDS):
        return "table_qa"

    if _contains_any(normalized, _SUMMARY_KEYWORDS):
        return "summarization"

    if "这个" in normalized or "那个" in normalized:
        if not _YEAR_RE.search(normalized):
            return "ambiguous_query"

    return normalize_query_type("fact_lookup")
