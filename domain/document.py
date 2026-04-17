"""Document domain models for trusted PDF QA indexing and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseModel):
    doc_id: str
    collection_name: str
    doc_source: str
    title: str = ""
    doc_hash: str = ""
    page_count: int = 0
    status: DocumentStatus = DocumentStatus.INDEXED
    metadata: Dict[str, Any] = Field(default_factory=dict)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
