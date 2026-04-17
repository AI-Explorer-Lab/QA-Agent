from __future__ import annotations

from typing import List

from utils.content_normalizer import normalize_whitespace


def expand_queries(
    question: str,
    query_type: str,
    expand_query_num: int = 3,
    retry_reason: str = "",
) -> List[str]:
    normalized = normalize_whitespace(question, preserve_newlines=False)
    max_expand = max(1, int(expand_query_num))

    expansions = [normalized]
    templates_by_type = {
        "fact_lookup": [
            "{q} 关键条款",
            "{q} 相关定义",
            "{q} 对应章节",
        ],
        "table_qa": [
            "{q} 指标 数值 单位 期间 来源",
            "{q} 财务表格",
            "{q} 表头 字段",
        ],
        "summarization": [
            "{q} 核心要点总结",
            "{q} 章节摘要",
            "{q} 主要结论",
        ],
        "citation_locate": [
            "{q} 原文片段 页码",
            "{q} 标题路径 chunk_id",
            "{q} 引用定位",
        ],
        "report_generation": [
            "{q} 报告提纲",
            "{q} 风险与机会",
            "{q} 关键数据引用",
        ],
        "multi_doc_compare": [
            "{q} 差异点 对照",
            "{q} 多文档 逐项比较",
            "{q} 版本变化",
        ],
        "ambiguous_query": [
            "{q} 需要补充信息",
        ],
    }

    for template in templates_by_type.get(query_type, []):
        if len(expansions) >= max_expand:
            break
        expansions.append(template.format(q=normalized))

    if retry_reason and len(expansions) < max_expand:
        expansions.append(f"{normalized} {retry_reason}")

    # Stable de-duplication while preserving order.
    deduped: List[str] = []
    seen = set()
    for item in expansions:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:max_expand]
