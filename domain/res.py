"""Response models for API entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Union

from pydantic import BaseModel, Field

from domain.citation import Citation, Evidence
from domain.models import Decision
from domain.retrieval import RetrievalTrace


class QAResponse(BaseModel):
    answer: str
    decision: Decision
    query_type: str
    confidence: float
    citations: List[Citation] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    retrieval_trace: Union[RetrievalTrace, Dict[str, object]] = Field(default_factory=dict)
    rerank_trace: Dict[str, object] = Field(default_factory=dict)
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
