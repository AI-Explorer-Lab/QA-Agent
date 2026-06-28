"""Dynaconf-backed application configuration entry point."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency bootstrap fallback
    load_dotenv = None  # type: ignore[assignment]

try:
    from dynaconf import Dynaconf
except Exception:  # pragma: no cover - dependency bootstrap fallback
    Dynaconf = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {_normalize_key(key): _normalize_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_mapping(item) for item in value]
    return value


def selected_environment() -> str:
    return (
        os.getenv("APP_ENV")
        or os.getenv("ENV_FOR_DYNACONF")
        or os.getenv("TRUSTED_QA_ENV")
        or "dev"
    ).strip() or "dev"


def build_settings(config_path: str | Path | None = None) -> Any:
    if Dynaconf is None:
        raise RuntimeError("dynaconf is not installed. Run `pip install -r requirements.txt`.")
    if load_dotenv is not None and DEFAULT_DOTENV_PATH.exists():
        load_dotenv(DEFAULT_DOTENV_PATH, override=False, encoding="utf-8-sig")
    path = Path(config_path or os.getenv("APP_CONFIG_PATH") or DEFAULT_CONFIG_PATH).expanduser()
    return Dynaconf(
        settings_files=[str(path)],
        environments=False,
        load_dotenv=False,
        envvar_prefix="TRUSTED_QA",
        lowercase_read=True,
        merge_enabled=True,
    )


def load_app_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.getenv("APP_CONFIG_PATH") or DEFAULT_CONFIG_PATH).expanduser()
    if Dynaconf is None:
        if yaml is None or not path.exists():
            data: Any = {}
        else:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        settings = build_settings(path)
        data = settings.as_dict()
    data = _normalize_mapping(data)
    if not isinstance(data, dict):
        return {}

    result = copy.deepcopy(data)
    result.setdefault("environment", {})
    if isinstance(result["environment"], dict):
        result["environment"].setdefault("name", selected_environment())
    return result
