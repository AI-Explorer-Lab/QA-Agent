from __future__ import annotations

from typing import Any

__all__ = [
    "DocumentIndexService",
    "TrustedQAWorkflow",
    "get_document_index_service",
    "get_trusted_qa_workflow",
]


def __getattr__(name: str) -> Any:
    if name in {"DocumentIndexService", "get_document_index_service"}:
        from service.agent.document_indexer import DocumentIndexService, get_document_index_service

        return {"DocumentIndexService": DocumentIndexService, "get_document_index_service": get_document_index_service}[name]
    if name in {"TrustedQAWorkflow", "get_trusted_qa_workflow"}:
        from service.agent.trusted_qa_workflow import TrustedQAWorkflow, get_trusted_qa_workflow

        return {"TrustedQAWorkflow": TrustedQAWorkflow, "get_trusted_qa_workflow": get_trusted_qa_workflow}[name]
    raise AttributeError(f"module 'service.agent' has no attribute {name!r}")