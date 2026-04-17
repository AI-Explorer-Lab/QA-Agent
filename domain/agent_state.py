"""Agent runtime state model for ReAct + skills orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from domain.citation import Citation, Evidence
from domain.retrieval import RetrievalCandidate, RetrievalTrace


class AgentPhase(str, Enum):
    LOAD_SESSION = "load_session"
    CLASSIFY_QUERY_TYPE = "classify_query_type"
    CLARIFY_GATE = "clarify_gate"
    SELECT_SKILL = "select_skill"
    RETRIEVAL = "parallel_hybrid_retrieval"
    RERANK = "two_stage_hybrid_rerank"
    EVIDENCE_GATE = "evidence_gate"
    ANSWER = "answer_with_citations"
    SAVE = "save_session"
    END = "end"


class AgentState(BaseModel):
    session_id: str
    collection_name: str
    question: str
    query_type: str = "ambiguous_query"

    phase: AgentPhase = AgentPhase.LOAD_SESSION
    selected_skill: str = ""

    top_k: int = 5
    expand_query_num: int = 3
    enable_cache: bool = True

    retries: int = 0
    max_retries: int = 2

    missing_slots: List[str] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)

    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    retrieval_trace: Optional[RetrievalTrace] = None

    evidence: List[Evidence] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    answer: str = ""
    decision: str = "answer"
    confidence: float = 0.0

    metadata: Dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
