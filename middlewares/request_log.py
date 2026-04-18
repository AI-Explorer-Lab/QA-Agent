from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

from middlewares.operation_log import configure_logging, log_operation_event
from middlewares.trace_context import (
    reset_request_id,
    reset_trace_id,
    set_request_id,
    set_trace_id,
)


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        trace_id = request.headers.get("x-trace-id") or request_id
        request_token = set_request_id(request_id)
        trace_token = set_trace_id(trace_id)
        started = time.perf_counter()
        configure_logging()
        log_operation_event(
            "http.request",
            status="started",
            method=request.method,
            path=str(request.url.path),
            client=str(request.client.host) if request.client else "",
        )
        try:
            response = await call_next(request)
            duration_ms = int((time.perf_counter() - started) * 1000)
            response.headers["x-request-id"] = request_id
            response.headers["x-trace-id"] = trace_id
            response.headers["x-process-time-ms"] = str(duration_ms)
            log_operation_event(
                "http.request",
                status="completed",
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            log_operation_event(
                "http.request",
                status="failed",
                level=logging.ERROR,
                method=request.method,
                path=str(request.url.path),
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            reset_trace_id(trace_token)
            reset_request_id(request_token)
