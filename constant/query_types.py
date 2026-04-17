"""Query type definitions used by classifier, skills, and response schema."""

from __future__ import annotations

from enum import Enum


class QueryType(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    TABLE_QA = "table_qa"
    SUMMARIZATION = "summarization"
    CITATION_LOCATE = "citation_locate"
    REPORT_GENERATION = "report_generation"
    MULTI_DOC_COMPARE = "multi_doc_compare"
    AMBIGUOUS_QUERY = "ambiguous_query"


SUPPORTED_QUERY_TYPES = tuple(query_type.value for query_type in QueryType)
