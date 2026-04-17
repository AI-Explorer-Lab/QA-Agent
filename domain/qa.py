"""Request/response/session models for trusted QA APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from domain.citation import Citation, Evidence
from domain.retrieval import RetrievalTrace


class Decision(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class QARequest(BaseModel):
    question: str
    session_id: str = ""
    collection_name: str
    top_k: int = 5
    expand_query_num: int = 3
    enable_cache: bool = True


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
