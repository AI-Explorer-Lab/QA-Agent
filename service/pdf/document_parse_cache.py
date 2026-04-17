from __future__ import annotations

from typing import Any, Dict

from service.embedding.embedding_cache import TTLCache


class DocumentParseCache(TTLCache[Dict[str, Any]]):
    pass
