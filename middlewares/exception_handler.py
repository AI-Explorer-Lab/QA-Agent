from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from middlewares.operation_log import log_operation_event

try:
    from exception import AppBaseException
except Exception:
    AppBaseException = None


async def app_exception_handler(request: Request, exc: Exception):
    if AppBaseException is not None and isinstance(exc, AppBaseException):
        log_operation_event(
            "http.exception",
            status="handled",
            level=logging.WARNING,
            path=str(request.url.path),
            status_code=exc.status_code,
            error_code=exc.code,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
    log_operation_event(
        "http.exception",
        status="unhandled",
        level=logging.ERROR,
        path=str(request.url.path),
        status_code=500,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": str(exc), "path": str(request.url.path)},
    )
