"""Request models for API entry points."""

from __future__ import annotations

from pydantic import BaseModel


class QARequest(BaseModel):
    question: str
    session_id: str = ""
    collection_name: str
    top_k: int = 5
    expand_query_num: int = 3
    enable_cache: bool = True
    use_llm_intent_slot: bool = False
    include_debug: bool = False


class DocumentIndexRequest(BaseModel):
    doc_source: str = ""
    pdf_path: str
    force_rebuild: bool = False
    collection_name: str = "default"
