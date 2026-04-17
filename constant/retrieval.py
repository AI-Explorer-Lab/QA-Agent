"""Retrieval strategy and scoring constants for the trusted QA agent."""

from __future__ import annotations

HYBRID_RETRIEVAL = "hybrid_retrieval"

DEFAULT_TOP_K = 5
DEFAULT_EXPAND_QUERY_NUM = 3
DEFAULT_MAX_CONCURRENCY = 6
DEFAULT_QUERY_TIMEOUT_SECONDS = 20
DEFAULT_TABLE_EVIDENCE_QUOTA = 2

DEFAULT_RERANK_WEIGHTS = {
    "dense_weight": 0.50,
    "bm25_weight": 0.35,
    "metadata_boost_weight": 0.10,
    "table_boost_weight": 0.05,
}

RETRIEVAL_SOURCES = (
    "dense",
    "bm25",
    "metadata",
    "table",
)
