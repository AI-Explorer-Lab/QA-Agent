from __future__ import annotations

import re

from service.agent.schemas import normalize_query_type
from utils.content_normalizer import normalize_whitespace

_TABLE_KEYWORDS = {
    "表格",
    "指标",
    "数据",
    "数值",
    "同比",
    "环比",
    "毛利率",
    "营业收入",
    "营业成本",
    "净利润",
    "现金流量净额",
    "每股收益",
    "研发投入",
    "非经常性损益",
    "货币资金",
    "应收账款",
    "参数",
    "table",
    "metric",
}

_FACT_LOOKUP_KEYWORDS = {
    "中文名称",
    "中文简称",
    "法定代表人",
    "注册地址",
    "办公地址",
    "公司网址",
    "电子信箱",
    "股票简称",
    "股票代码",
    "上市板块",
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
    "生成报告",
    "撰写报告",
    "输出报告",
    "形成报告",
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

_FINANCIAL_TABLE_TERMS = {
    "营业收入",
    "营业成本",
    "主营业务收入",
    "归属于上市公司股东的净利润",
    "净利润",
    "现金流量净额",
    "经营活动产生的现金流量净额",
    "基本每股收益",
    "稀释每股收益",
    "每股收益",
    "研发投入",
    "研发投入合计",
    "研发投入总额占营业收入比例",
    "非经常性损益",
    "委托他人投资或管理资产的损益",
    "货币资金",
    "应收账款",
    "账面价值",
    "期末余额",
    "经销",
    "直销",
}

_VALUE_QUERY_HINTS = {
    "多少",
    "分别",
    "合计",
    "金额",
    "余额",
    "比例",
    "同比",
    "变动比例",
    "期末",
    "年末",
    "分季度",
    "分类",
    "分解",
}

_EXPLANATION_HINTS = {
    "原因",
    "为什么",
    "影响",
    "说明",
}


def _contains_any(text: str, words: set[str]) -> bool:
    return any(word in text for word in words)


def is_financial_table_query(question: str) -> bool:
    normalized = normalize_whitespace(question, preserve_newlines=False).lower()
    if not normalized:
        return False

    has_metric = _contains_any(normalized, _FINANCIAL_TABLE_TERMS) or _contains_any(normalized, _TABLE_KEYWORDS)
    has_value_intent = _contains_any(normalized, _VALUE_QUERY_HINTS) or bool(_YEAR_RE.search(normalized))
    asks_explanation = _contains_any(normalized, _EXPLANATION_HINTS)

    if asks_explanation and "多少" not in normalized and "分别" not in normalized:
        return False
    return has_metric and has_value_intent


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

    if _contains_any(normalized, _FACT_LOOKUP_KEYWORDS):
        return "fact_lookup"

    if is_financial_table_query(normalized):
        return "table_qa"

    if _contains_any(normalized, _SUMMARY_KEYWORDS):
        return "summarization"

    if "这个" in normalized or "那个" in normalized:
        if not _YEAR_RE.search(normalized):
            return "ambiguous_query"

    return normalize_query_type("fact_lookup")
