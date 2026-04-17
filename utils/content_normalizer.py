from __future__ import annotations

import json
import re
from typing import Any, Iterable, List

_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: Any, preserve_newlines: bool = True) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    value = (
        raw.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00A0", " ")
        .replace("\u3000", " ")
    )

    if preserve_newlines:
        normalized_lines = [_INLINE_WHITESPACE_RE.sub(" ", line).strip() for line in value.split("\n")]
        value = "\n".join(line for line in normalized_lines if line)
        value = _MULTI_NEWLINE_RE.sub("\n\n", value)
        return value.strip()

    value = _INLINE_WHITESPACE_RE.sub(" ", value)
    return value.replace("\n", " ").strip()


def _normalize_list_content(items: Iterable[Any]) -> str:
    parts: List[str] = []
    for item in items:
        normalized = normalize_content(item)
        if normalized:
            parts.append(normalized)
    return normalize_whitespace("\n".join(parts), preserve_newlines=True)


def normalize_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return normalize_whitespace(content, preserve_newlines=True)

    if isinstance(content, list):
        return _normalize_list_content(content)

    if isinstance(content, dict):
        for key in ("text", "output_text", "value"):
            value = content.get(key)
            if isinstance(value, str):
                return normalize_whitespace(value, preserve_newlines=True)

        if "content" in content:
            return normalize_content(content.get("content"))

        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(content)

    return normalize_whitespace(content, preserve_newlines=True)
