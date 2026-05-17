from __future__ import annotations

import re
from typing import List

from utils.content_normalizer import normalize_whitespace


FIXED_QUERY_VARIANT_TOTAL = 4


def expand_queries(
    question: str,
    query_type: str,
    expand_query_num: int = 3,
    retry_reason: str = "",
) -> List[str]:
    del expand_query_num
    normalized = normalize_whitespace(question, preserve_newlines=False)
    concise = _concise_rewrite(normalized)
    compact = _compact_rewrite(normalized)
    scenario = _scenario_enhanced_query(concise or normalized, query_type)

    expansions = [
        normalized,
        concise or normalized,
        compact or concise or normalized,
    ]
    if retry_reason:
        scenario = normalize_whitespace(f"{scenario} {retry_reason}", preserve_newlines=False)

    deduped: List[str] = []
    seen = set()
    for item in expansions:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    fillers = [
        normalize_whitespace(f"{normalized} \u6838\u5fc3\u5173\u952e\u8bcd", preserve_newlines=False),
        normalize_whitespace(f"{normalized} \u76f8\u5173\u5185\u5bb9", preserve_newlines=False),
    ]
    for item in fillers:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
        if len(deduped) >= FIXED_QUERY_VARIANT_TOTAL - 1:
            break

    scenario_key = scenario.strip().lower()
    if scenario_key and scenario_key in seen:
        scenario = _scenario_enhanced_query(normalized, query_type)
        scenario_key = scenario.strip().lower()
    if scenario_key and scenario_key not in seen:
        deduped.append(scenario)

    return deduped[:FIXED_QUERY_VARIANT_TOTAL]


def _concise_rewrite(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""

    compact = text
    for pattern in [
        r"\u8bf7\u95ee",
        r"\u9ebb\u70e6",
        r"\u5e2e\u6211",
        r"\u5e2e\u5fd9",
        r"\u80fd\u4e0d\u80fd",
        r"\u662f\u5426\u53ef\u4ee5",
        r"\u53ef\u4ee5\u5417",
        r"\u8fd9\u4e2a\u6587\u6863(\u91cc|\u4e2d)?",
        r"\u8be5\u6587\u6863(\u91cc|\u4e2d)?",
        r"\u6839\u636e(\u6587\u6863|\u6750\u6599|\u8d44\u6599)",
        r"\u8bf7(\u7ed9\u51fa|\u8bf4\u660e|\u67e5\u8be2|\u67e5\u627e|\u56de\u7b54)",
        r"\u662f\u4ec0\u4e48",
        r"\u662f\u591a\u5c11",
        r"\u6709\u54ea\u4e9b",
        r"\u5982\u4f55",
    ]:
        compact = re.sub(pattern, " ", compact, flags=re.IGNORECASE)
    compact = re.sub(r"[\?\uff1f!\uff01,\uff0c\u3002\uff1b;\uff1a:\u3001]+", " ", compact)
    compact = normalize_whitespace(compact, preserve_newlines=False)
    return compact or text


def _compact_rewrite(question: str) -> str:
    concise = _concise_rewrite(question)
    if not concise:
        return ""

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-/.%]*|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}", concise)
    stopwords = {
        "\u4ec0\u4e48",
        "\u591a\u5c11",
        "\u54ea\u4e9b",
        "\u600e\u4e48",
        "\u5982\u4f55",
        "\u8bf7\u7ed9",
        "\u7ed9\u51fa",
        "\u8bf4\u660e",
        "\u67e5\u8be2",
        "\u67e5\u627e",
        "\u76f8\u5173",
        "\u5185\u5bb9",
        "\u6587\u6863",
        "\u6750\u6599",
    }
    kept = [token for token in tokens if token not in stopwords]
    return normalize_whitespace(" ".join(kept), preserve_newlines=False) or concise


def _scenario_enhanced_query(question: str, query_type: str) -> str:
    hints_by_type = {
        "fact_lookup": "\u5b9a\u4e49 \u5bf9\u5e94\u7ae0\u8282",
        "table_qa": "\u6307\u6807 \u6570\u503c \u5355\u4f4d \u8868\u5934",
        "citation_locate": "\u539f\u6587\u51fa\u5904 \u6807\u9898\u8def\u5f84 \u7ae0\u8282 \u539f\u6587\u7247\u6bb5",
    }
    hints = hints_by_type.get(str(query_type or "fact_lookup"), "\u5b9a\u4e49 \u5bf9\u5e94\u7ae0\u8282")
    return normalize_whitespace(f"{question} {hints}", preserve_newlines=False)
