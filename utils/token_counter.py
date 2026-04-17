from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

from utils.content_normalizer import normalize_whitespace

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency at runtime
    tiktoken = None

_ENCODING_CACHE: Dict[str, object] = {}


def _fallback_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def _get_encoding(encoding_name: str):
    if tiktoken is None:
        return None

    cached = _ENCODING_CACHE.get(encoding_name)
    if cached is not None:
        return cached

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        _ENCODING_CACHE[encoding_name] = encoding
        return encoding
    except Exception:
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    normalized = normalize_whitespace(text, preserve_newlines=True)
    if not normalized:
        return 0

    encoding = _get_encoding(encoding_name)
    if encoding is None:
        return _fallback_token_count(normalized)

    try:
        return len(encoding.encode(normalized))
    except Exception:
        return _fallback_token_count(normalized)


def count_tokens_batch(texts: Iterable[str], encoding_name: str = "cl100k_base") -> int:
    return sum(count_tokens(text, encoding_name=encoding_name) for text in texts)


def truncate_to_token_limit(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    max_tokens = max(0, int(max_tokens))
    normalized = normalize_whitespace(text, preserve_newlines=True)
    if not normalized or max_tokens <= 0:
        return ""

    encoding = _get_encoding(encoding_name)
    if encoding is None:
        approx_chars = max_tokens * 4
        return normalized[:approx_chars].strip()

    try:
        tokens = encoding.encode(normalized)
    except Exception:
        approx_chars = max_tokens * 4
        return normalized[:approx_chars].strip()

    if len(tokens) <= max_tokens:
        return normalized

    try:
        return normalize_whitespace(encoding.decode(tokens[:max_tokens]), preserve_newlines=True)
    except Exception:
        approx_chars = max_tokens * 4
        return normalized[:approx_chars].strip()
