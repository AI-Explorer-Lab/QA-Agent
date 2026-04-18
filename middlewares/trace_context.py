from __future__ import annotations

import contextvars
from contextvars import Token

trace_id_var = contextvars.ContextVar("trace_id", default="")
request_id_var = contextvars.ContextVar("request_id", default="")


def get_trace_id() -> str:
    return trace_id_var.get()


def get_request_id() -> str:
    return request_id_var.get()


def set_trace_id(trace_id: str) -> Token:
    return trace_id_var.set(trace_id)


def set_request_id(request_id: str) -> Token:
    return request_id_var.set(request_id)


def reset_trace_id(token: Token) -> None:
    trace_id_var.reset(token)


def reset_request_id(token: Token) -> None:
    request_id_var.reset(token)
