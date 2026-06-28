"""Business exceptions used by domain and mapper layers."""

from __future__ import annotations

from constant import (
    DOCUMENT_NOT_FOUND,
    LOW_EVIDENCE,
    NOT_FOUND,
    RETRIEVAL_ERROR,
    SESSION_NOT_FOUND,
    VALIDATION_ERROR,
)
from exceptions.base_exception import AppBaseException


class ValidationException(AppBaseException):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message=message, code=VALIDATION_ERROR, status_code=400, detail=detail or {})


class DocumentNotFoundException(AppBaseException):
    def __init__(self, doc_id: str) -> None:
        super().__init__(
            message=f"Document not found: {doc_id}",
            code=DOCUMENT_NOT_FOUND,
            status_code=404,
            detail={"doc_id": doc_id},
        )


class CollectionNotFoundException(AppBaseException):
    def __init__(self, collection_name: str) -> None:
        super().__init__(
            message=f"Collection not found: {collection_name}",
            code=NOT_FOUND,
            status_code=404,
            detail={"collection_name": collection_name},
        )


class SessionNotFoundException(AppBaseException):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session not found: {session_id}",
            code=SESSION_NOT_FOUND,
            status_code=404,
            detail={"session_id": session_id},
        )


class RetrievalException(AppBaseException):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message=message, code=RETRIEVAL_ERROR, status_code=422, detail=detail or {})


class LowEvidenceException(AppBaseException):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message=message, code=LOW_EVIDENCE, status_code=422, detail=detail or {})

