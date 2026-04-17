from __future__ import annotations

QUERY_TYPES = (
    "fact_lookup",
    "table_qa",
    "summarization",
    "citation_locate",
    "report_generation",
    "multi_doc_compare",
    "ambiguous_query",
)

QUERY_TYPE_SET = set(QUERY_TYPES)

FINAL_DECISIONS = {"answer", "clarify", "refuse"}


def normalize_query_type(query_type: str | None) -> str:
    value = str(query_type or "").strip().lower()
    if value in QUERY_TYPE_SET:
        return value
    return "fact_lookup"
