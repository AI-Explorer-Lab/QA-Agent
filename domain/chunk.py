"""Chunk domain models for structured PDF text/table retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    collection_name: str
    doc_source: str

    page_idx: Optional[int] = None
    page_range: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_index: int = 0

    heading_path: str = ""
    level1_title: str = ""
    level2_title: str = ""
    level3_title: str = ""

    table_id: str = ""
    sub_table_id: str = ""
    table_header_text: str = ""
    table_context_text: str = ""

    content: str
    search_text: str = ""
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
