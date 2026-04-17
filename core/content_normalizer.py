from __future__ import annotations

import json
from typing import Any, List


def _normalize_list_content(items: List[Any]) -> str:
    parts: List[str] = []
    for item in items:
        normalized = normalize_content(item)
        if normalized:
            parts.append(normalized)
    if parts:
        return "\n".join(part for part in parts if part).strip()
    try:
        return json.dumps(items, ensure_ascii=False)
    except Exception:
        return str(items)


def normalize_content(content: Any) -> str:
    """
    Convert LLM message content into plain text.

    Compatible with both classic ChatCompletions string content and
    Responses API structured list/dict content blocks.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return _normalize_list_content(content)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

        nested = content.get("content")
        if nested is not None:
            return normalize_content(nested)

        output = content.get("output_text")
        if isinstance(output, str):
            return output

        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    return str(content)
