from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from middlewares.trace_context import get_request_id, get_trace_id

LOGGER_NAME = "trusted_qa.operation"
_LOGGING_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("trusted_qa").setLevel(level)
    _LOGGING_CONFIGURED = True


def _compact_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def log_operation_event(
    event: str,
    status: str = "info",
    level: int = logging.INFO,
    message: str = "",
    **fields: Any,
) -> None:
    configure_logging()
    payload = {
        "event": event,
        "status": status,
        "trace_id": get_trace_id(),
        "request_id": get_request_id(),
    }
    if message:
        payload["message"] = message
    payload.update(_compact_fields(fields))
    logging.getLogger(LOGGER_NAME).log(
        level,
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True),
    )


@dataclass
class OperationTimer:
    event: str
    fields: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)


def start_operation_step(event: str, **fields: Any) -> OperationTimer:
    timer = OperationTimer(event=event, fields=_compact_fields(fields))
    log_operation_event(event, status="started", **timer.fields)
    return timer


def finish_operation_step(
    timer: OperationTimer,
    status: str = "completed",
    level: int = logging.INFO,
    message: str = "",
    **fields: Any,
) -> None:
    payload = dict(timer.fields)
    payload.update(_compact_fields(fields))
    payload["duration_ms"] = timer.elapsed_ms()
    log_operation_event(
        timer.event,
        status=status,
        level=level,
        message=message,
        **payload,
    )


def fail_operation_step(timer: OperationTimer, exc: Exception, **fields: Any) -> None:
    finish_operation_step(
        timer,
        status="failed",
        level=logging.ERROR,
        error_type=type(exc).__name__,
        error=str(exc),
        **fields,
    )
