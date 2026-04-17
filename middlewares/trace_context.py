from __future__ import annotations

import contextvars

trace_id_var = contextvars.ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)
