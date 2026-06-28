"""Shared domain models that are neither API-only requests nor responses."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from domain.citation import Citation, Evidence


class Decision(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class QASession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    collection_name: str
    user_id: str = ""
    metadata: Dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QAMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: MessageRole
    query_type: str = ""

    question: str = ""
    answer: str = ""
    decision: Optional[Decision] = None
    confidence: float = 0.0

    citations: List[Citation] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    retrieval_trace_id: str = ""
    metadata: Dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
