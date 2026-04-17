from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _to_hash_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return str(value)

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(_to_hash_text(value).encode("utf-8")).hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    size = max(4, int(length))
    return stable_sha256(value)[:size]


def build_cache_key(*parts: Any, prefix: str = "") -> str:
    joined = "\x1f".join(_to_hash_text(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    if prefix:
        return f"{prefix}:{digest}"
    return digest


def file_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
