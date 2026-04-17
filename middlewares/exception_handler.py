from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

try:
    from exception import AppBaseException
except Exception:
    AppBaseException = None


async def app_exception_handler(request: Request, exc: Exception):
    if AppBaseException is not None and isinstance(exc, AppBaseException):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": str(exc), "path": str(request.url.path)},
    )
