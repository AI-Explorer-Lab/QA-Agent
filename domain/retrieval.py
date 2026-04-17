"""Retrieval candidate and trace models for hybrid retrieval workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    doc_id: str
    collection_name: str
    doc_source: str
    content: str

    query: str = ""
    query_type: str = "fact_lookup"
    retrieval_source: str = "dense"

    chunk_type: str = "text"
    page_idx: Optional[int] = None
    page_range: str = ""
    heading_path: str = ""

    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata_boost: float = 0.0
    table_boost: float = 0.0
    final_score: float = 0.0

    metadata: Dict[str, object] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    message_id: str = ""
    collection_name: str = ""

    question: str
    query_type: str = "fact_lookup"
    expanded_queries: List[str] = Field(default_factory=list)

    dense_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    sparse_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    merged_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    selected_candidates: List[RetrievalCandidate] = Field(default_factory=list)

    rerank_trace: Dict[str, object] = Field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
