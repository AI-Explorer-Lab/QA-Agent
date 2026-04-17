"""Base exception types for the trusted PDF QA project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from constant import INTERNAL_ERROR


@dataclass
class AppBaseException(Exception):
    message: str
    code: str = INTERNAL_ERROR
    status_code: int = 400
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "status_code": self.status_code,
        }


class ConfigException(AppBaseException):
    pass


class DatabaseException(AppBaseException):
    pass
