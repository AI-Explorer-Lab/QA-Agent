"""Citation and evidence models for grounded answer generation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    doc_id: str
    doc_source: str
    collection_name: str
    page_idx: Optional[int] = None
    page_range: str = ""
    heading_path: str = ""
    quote: str = ""
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    confidence: float = 0.0


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    doc_id: str
    doc_source: str
    chunk_type: str = "text"
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)
    citation: Optional[Citation] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
