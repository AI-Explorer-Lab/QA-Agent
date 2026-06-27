"""Application-level route constants for the trusted PDF QA API."""

HEALTH_ROUTE = "/health"
DOCUMENT_INDEX_ROUTE = "/documents/index"
QA_ASK_ROUTE = "/qa/ask"
QA_ASK_STREAM_ROUTE = "/qa/ask/stream"
QA_SESSION_ROUTE = "/qa/sessions/{session_id}"

ALL_ROUTES = (
    HEALTH_ROUTE,
    DOCUMENT_INDEX_ROUTE,
    QA_ASK_ROUTE,
    QA_ASK_STREAM_ROUTE,
    QA_SESSION_ROUTE,
)
